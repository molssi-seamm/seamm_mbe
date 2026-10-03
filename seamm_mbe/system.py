"""The system: atoms, cell, whole molecules and their types.

A :class:`System` is built from arrays (symbols, Cartesian coordinates in Å,
the cell, the bonds) or from a molsystem configuration
(:meth:`System.from_configuration`). It finds the molecules as the connected
components of the bond graph, makes each molecule whole, types the molecules
by formula and topology, and settles each type's charge and multiplicity.

Molecules are numbered in the order of their first atom, and each molecule's
atoms are in ascending atom order. A molecule's *home* coordinates keep its
first atom where it is and place the others by following the bonds with the
minimum image, so a molecule that is already whole keeps exactly the
coordinates it was given.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .catalog import CATALOG, MoleculeType, signature, wl_labels
from .elements import IONIC_ELEMENTS, atomic_number, hill_formula, mass


class StructureError(ValueError):
    """The structure cannot be used as given (bad charges, missing bonds, ...)."""


@dataclass
class Molecule:
    """One whole molecule.

    Attributes
    ----------
    index : int
        0-based molecule number.
    atoms : numpy.ndarray of int
        The atoms (0-based indices in the system), ascending.
    type : str
        The name of its :class:`~seamm_mbe.catalog.MoleculeType`.
    coordinates : numpy.ndarray
        (n, 3) home coordinates in Å, whole.
    designated : int
        Index into ``atoms`` of the designated atom.
    """

    index: int
    atoms: np.ndarray
    type: str
    coordinates: np.ndarray
    designated: int


class System:
    """Atoms, an optional cell, and the molecules.

    Parameters
    ----------
    symbols : [str]
        Element symbols.
    coordinates : array-like
        (n, 3) Cartesian coordinates in Å.
    cell : array-like or None
        (3, 3) lattice vectors in Å, one per **row**; None for a cluster.
    bonds : [(int, int)]
        Bonds as 0-based atom index pairs. They define the molecules; atoms
        with no bonds are one-atom molecules (ions).
    formal_charges : [int] or None
        Per-atom formal charges, e.g. from an SDF file's ``M  CHG`` lines. When
        given and not all zero they define each molecule's charge, and a
        catalog type whose charge disagrees is an error. When absent or all
        zero (bare xyz input) the catalog's charges are used.
    catalog : {str: MoleculeType} or None
        Known types keyed by signature; default :data:`seamm_mbe.CATALOG`.
    charges : {str: int} or None
        Charges by type name, overriding both the formal charges and the
        catalog.
    multiplicities : {str: int} or None
        Multiplicities by type name, overriding the catalog (default 1).
    masses : [float] or None
        Atomic masses (g/mol); default the standard atomic weights.
    """

    def __init__(
        self,
        symbols,
        coordinates,
        cell=None,
        *,
        bonds=(),
        formal_charges=None,
        catalog=None,
        charges=None,
        multiplicities=None,
        masses=None,
    ):
        self.symbols = list(symbols)
        self.coordinates = np.array(coordinates, dtype=float).reshape(-1, 3)
        if len(self.symbols) != len(self.coordinates):
            raise StructureError(
                f"{len(self.symbols)} symbols but {len(self.coordinates)} coordinates"
            )
        self.cell = None if cell is None else np.array(cell, dtype=float)
        if self.cell is not None and self.cell.shape != (3, 3):
            raise StructureError(
                "The cell must be 3 lattice vectors (3x3, one per row)"
            )
        self.atomic_numbers = np.array([atomic_number(s) for s in self.symbols])
        self.masses = np.array(
            [mass(s) for s in self.symbols] if masses is None else masses, dtype=float
        )
        self.bonds = [(int(i), int(j)) for i, j in bonds]
        self.catalog = CATALOG if catalog is None else catalog
        self.types = {}
        self.molecules = []
        self._find_molecules()
        self._type_molecules()
        self._set_charges(formal_charges, charges or {}, multiplicities or {})

    # ----------------------------------------------------------------- geometry
    @property
    def periodic(self):
        """Whether the system is a periodic cell (False for a cluster)."""
        return self.cell is not None

    @property
    def volume(self):
        """The cell volume in Å³ (None for a cluster)."""
        return None if self.cell is None else float(abs(np.linalg.det(self.cell)))

    @property
    def widths(self):
        """The three perpendicular widths of the cell (Å): the spacing of the
        planes spanned by each pair of lattice vectors. The smallest is
        L_min, the bound on any cutoff. None for a cluster."""
        if self.cell is None:
            return None
        a, b, c = self.cell
        volume = self.volume
        return np.array(
            [
                volume / np.linalg.norm(np.cross(b, c)),
                volume / np.linalg.norm(np.cross(c, a)),
                volume / np.linalg.norm(np.cross(a, b)),
            ]
        )

    def shift(self, image):
        """The Cartesian translation (Å) of an integer image (n_a, n_b, n_c)."""
        if self.cell is None:
            return np.zeros(3)
        return np.asarray(image, dtype=float) @ self.cell

    def minimum_image(self, vector):
        """The integer image n making ``vector + n @ cell`` shortest, and that
        shortest vector. Searches the 27 images around the fractional rounding,
        so it is exact for any cell shape. For a cluster, n = (0, 0, 0)."""
        vector = np.asarray(vector, dtype=float)
        if self.cell is None:
            return (0, 0, 0), vector
        fractional = np.linalg.solve(self.cell.T, vector)
        base = -np.round(fractional)
        shifts = base + _NEIGHBORS
        candidates = vector + shifts @ self.cell
        k = int(np.argmin(np.einsum("ij,ij->i", candidates, candidates)))
        return tuple(int(v) for v in shifts[k]), candidates[k]

    # ---------------------------------------------------------------- molecules
    def _find_molecules(self):
        n = len(self.symbols)
        neighbors = [[] for _ in range(n)]
        for i, j in self.bonds:
            if i == j:
                continue
            neighbors[i].append(j)
            neighbors[j].append(i)
        seen = np.full(n, False)
        self._neighbors = neighbors
        for first in range(n):
            if seen[first]:
                continue
            # Follow the bonds from the first atom, placing each atom at the
            # minimum image of its bond to the atom it was reached from.
            xyz = {first: self.coordinates[first]}
            queue = deque([first])
            seen[first] = True
            while queue:
                i = queue.popleft()
                for j in neighbors[i]:
                    if j in xyz:
                        continue
                    _, d = self.minimum_image(self.coordinates[j] - self.coordinates[i])
                    xyz[j] = xyz[i] + d
                    seen[j] = True
                    queue.append(j)
            atoms = np.array(sorted(xyz))
            self.molecules.append(
                Molecule(
                    index=len(self.molecules),
                    atoms=atoms,
                    type="",
                    coordinates=np.array([xyz[a] for a in atoms]),
                    designated=0,
                )
            )

    def _molecule_graph(self, molecule):
        local = {a: k for k, a in enumerate(molecule.atoms)}
        symbols = [self.symbols[a] for a in molecule.atoms]
        bonds = sorted(
            {
                (min(local[i], local[j]), max(local[i], local[j]))
                for i, j in self.bonds
                if i in local and j in local and i != j
            }
        )
        return symbols, bonds

    def _type_molecules(self):
        by_signature = {}
        for molecule in self.molecules:
            symbols, bonds = self._molecule_graph(molecule)
            sig = signature(symbols, bonds)
            labels = wl_labels(symbols, bonds)
            if sig in self.catalog:
                mtype = self.catalog[sig]
            elif sig in by_signature:
                mtype = by_signature[sig]
            else:
                formula = hill_formula(symbols)
                name = formula
                n = 1
                while name in self.types:
                    n += 1
                    name = f"{formula}#{n}"
                mtype = MoleculeType(
                    name=name,
                    formula=formula,
                    signature=sig,
                    charge=None,
                    designated=labels[self._central_atom(molecule)],
                )
            by_signature[sig] = mtype
            if mtype.name not in self.types:
                # A per-system copy: charges may be set from the structure
                self.types[mtype.name] = MoleculeType(**vars(mtype))
            molecule.type = mtype.name
            molecule.designated = labels.index(mtype.designated)

    def _central_atom(self, molecule):
        """The heavy atom (any atom if there is none) nearest the centre of
        mass: the default designated atom of an automatically typed molecule."""
        xyz = molecule.coordinates
        m = self.masses[molecule.atoms]
        com = m @ xyz / m.sum()
        heavy = [k for k, a in enumerate(molecule.atoms) if self.symbols[a] != "H"]
        candidates = heavy or list(range(len(molecule.atoms)))
        return min(candidates, key=lambda k: np.linalg.norm(xyz[k] - com))

    def _set_charges(self, formal_charges, charges, multiplicities):
        for name in list(charges) + list(multiplicities):
            if name not in self.types:
                raise StructureError(
                    f"Charge or multiplicity given for {name!r}, but there is no "
                    f"molecule of that type; the types are {sorted(self.types)}."
                )
        use_formal = formal_charges is not None and any(int(q) for q in formal_charges)
        if use_formal:
            formal = np.asarray(formal_charges, dtype=int)
        for molecule in self.molecules:
            mtype = self.types[molecule.type]
            if mtype.name in charges:
                charge = int(charges[mtype.name])
            elif use_formal:
                charge = int(formal[molecule.atoms].sum())
                if mtype.charge is not None and charge != mtype.charge:
                    raise StructureError(
                        f"Molecule {molecule.index} ({mtype.name}, atoms "
                        f"{_atom_list(molecule.atoms)}) has formal charges summing "
                        f"to {charge:+d}, but a {mtype.name} has charge "
                        f"{mtype.charge:+d}. Fix the structure's formal charges or "
                        "give the charge of the type explicitly."
                    )
            else:
                charge = 0 if mtype.charge is None else mtype.charge
            if mtype.charge is None or mtype.name in charges:
                mtype.charge = charge
            elif charge != mtype.charge:
                raise StructureError(
                    f"Molecule {molecule.index} ({mtype.name}) has charge {charge:+d}"
                    f" but another {mtype.name} has {mtype.charge:+d}; molecules "
                    "of one type must have the same charge."
                )
            if mtype.name in multiplicities:
                mtype.multiplicity = int(multiplicities[mtype.name])
        for mtype in self.types.values():
            molecule = next(m for m in self.molecules if m.type == mtype.name)
            self._check_parity(molecule, mtype)

    def _check_parity(self, molecule, mtype):
        """Electron-count parity of a molecule at its type's charge and
        multiplicity -- the same check seamm_bsse makes for fragments."""
        n_electrons = int(self.atomic_numbers[molecule.atoms].sum()) - mtype.charge
        if (n_electrons + mtype.multiplicity - 1) % 2 == 0:
            return
        if mtype.multiplicity == 1:
            need = "a closed-shell (singlet) molecule needs an even number"
        else:
            parity = "even" if mtype.multiplicity % 2 == 1 else "odd"
            need = f"multiplicity {mtype.multiplicity} needs an {parity} number"
        raise StructureError(
            f"Molecule {molecule.index!r} ({mtype.name}, {len(molecule.atoms)} "
            f"atom(s)) has charge {mtype.charge:+d}, giving it {n_electrons} "
            f"electrons; {need}. Check that the charges are assigned to the "
            "molecules they actually belong to -- the structure's formal charges, "
            "or the charges given by type."
        )

    # ------------------------------------------------------------------ helpers
    def reference_point(self, molecule, criterion):
        """The point (Å, home coordinates) that locates a molecule for a
        distance criterion: "designated", "com" or "cog". The contact criteria
        use the centre of geometry as their reference point."""
        xyz = molecule.coordinates
        if criterion == "designated":
            return xyz[molecule.designated]
        if criterion == "com":
            m = self.masses[molecule.atoms]
            return m @ xyz / m.sum()
        return xyz.mean(axis=0)

    def formula(self):
        """The system's Hill formula."""
        return hill_formula(self.symbols)

    def type_counts(self):
        """{type name: number of molecules}."""
        counts = {}
        for molecule in self.molecules:
            counts[molecule.type] = counts.get(molecule.type, 0) + 1
        return counts

    @classmethod
    def from_configuration(cls, configuration, **kwargs):
        """A System from a molsystem configuration.

        The configuration must have its bonds (call its ``perceive_bonds()``
        first if it has none): they define the molecules. Per-atom formal
        charges, if the configuration carries them, set the molecules'
        charges (see :class:`System`). Keyword arguments are passed to
        :class:`System`.
        """
        atoms = configuration.atoms
        symbols = atoms.symbols
        if configuration.periodicity == 3:
            cell = np.array(configuration.cell.vectors())
        elif configuration.periodicity == 0:
            cell = None
        else:
            raise StructureError(
                f"Periodicity {configuration.periodicity} is not supported: only "
                "3-D cells and clusters."
            )
        xyz = np.array(atoms.get_coordinates(fractionals=False))
        neighbors = configuration.bonded_neighbors(as_indices=True)
        bonds = [(i, j) for i, js in enumerate(neighbors) for j in js if i < j]
        bondable = sum(1 for s in symbols if s not in IONIC_ELEMENTS)
        if not bonds and bondable > 1:
            raise StructureError(
                "The configuration has no bonds, so its molecules are unknown. "
                "Perceive the bonds first (configuration.perceive_bonds())."
            )
        if "formal_charges" not in kwargs and "formal_charge" in atoms:
            kwargs["formal_charges"] = [
                int(q or 0) for q in atoms.get_column_data("formal_charge")
            ]
        return cls(symbols, xyz, cell, bonds=bonds, **kwargs)


_NEIGHBORS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
)


def _atom_list(atoms, limit=6):
    text = ", ".join(str(a) for a in atoms[:limit])
    return text + (", ..." if len(atoms) > limit else "")
