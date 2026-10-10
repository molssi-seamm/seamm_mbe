"""Ion shells: an ion and its first-shell molecules as one unit of the expansion.

Around a strongly polarizing ion (Li⁺) the per-molecule expansion converges
slowly: in the C.3 cluster (Li⁺/BF₄⁻/6 H₂O/4 EC, 2026-10-09) the Li⁺ 3- and
4-body [revDSD − r2SCAN] increments summed to +19 to +23 and −5 to −18 kJ/mol.
With Li⁺ and its first shell as one unit the error at three units fell to
+1–2 kJ/mol and the forces to 3.6 meV/Å RMS (seamm_mbe campaign 2026-10-09).

:class:`IonShellRules` says which ions get a shell and how far it reaches.
:func:`ion_shells` returns a :class:`UnitSystem`, a view of a
:class:`~seamm_mbe.System` whose ``molecules`` are the units: each shell (the
ion and its members, placed whole around the ion) and every other molecule by
itself. :func:`~seamm_mbe.enumerate_fragments` works on it unchanged, so the
fragments are built from units. The real molecules stay on
:attr:`UnitSystem.parent`, for what is per molecule (energy offsets, the
molecular virial).

**Membership.** A molecule is a candidate for an ion's shell when one of its
atoms of a listed element lies within that element's cutoff of the ion (e.g.
Li–O 2.6 Å, Li–F 2.6 Å). Ions never join a shell (only molecules with listed
atoms can), and anions get none of their own, so a contact ion pair puts the
anion in the cation's shell. A molecule near two ions joins the nearer one
(``shared = "nearest"``), so units never overlap; with ``max_members`` a crowded
shell keeps its nearest members and the rest stay units of their own. The
units are rebuilt from each frame's geometry.
"""

from dataclasses import dataclass, field

import numpy as np

from .catalog import MoleculeType
from .elements import hill_formula
from .system import Molecule, StructureError, System


class ShellError(StructureError):
    """The ion-shell rules cannot be applied to this system."""


def _default_cutoffs():
    return {"Li": {"O": 2.6, "F": 2.6}}


@dataclass
class IonShellRules:
    """Which ions get a shell, and its reach.

    Attributes
    ----------
    cutoffs : {str: {str: float}}
        Per ion element, the shell cutoff (Å) to each partner element: a
        molecule joins the shell when one of its atoms of that element is
        within the cutoff of the ion. Default Li⁺ with O and F at 2.6 Å, just
        inside the first minima of g(r) in carbonate electrolytes.
    shared : str
        What happens to a molecule within reach of two ions: "nearest" joins
        the nearer one.
    max_members : int or None
        The most molecules in one shell (the nearest are kept); None for no
        limit.
    """

    cutoffs: dict = field(default_factory=_default_cutoffs)
    shared: str = "nearest"
    max_members: int | None = 5

    def __post_init__(self):
        if self.shared != "nearest":
            raise ShellError(
                f"Unknown handling of shared shell molecules {self.shared!r}: only "
                "'nearest' (merging two shells makes units too large to compute)."
            )
        if self.max_members is not None and int(self.max_members) < 1:
            raise ShellError("A shell must be allowed at least one member")
        for ion, partners in self.cutoffs.items():
            if not partners:
                raise ShellError(f"No partner elements for the {ion} shells")
            if any(float(c) <= 0 for c in partners.values()):
                raise ShellError(f"The {ion} shell cutoffs must be positive")


class UnitSystem:
    """A :class:`~seamm_mbe.System` seen as units: shells and other molecules.

    It has the attributes and methods of a System that fragment enumeration
    uses, with ``molecules`` and ``types`` describing the units. A unit's
    ``atoms`` are system atom indices, ascending, and its ``coordinates`` are
    whole around the ion.

    Attributes
    ----------
    parent : seamm_mbe.System
        The real system.
    members : [tuple of int]
        Per unit, the parent's molecules it holds (the ion first for a shell).
    shells : frozenset of int
        The units that are shells.
    """

    periodic = System.periodic
    volume = System.volume
    widths = System.widths
    masses = System.masses
    shift = System.shift
    minimum_image = System.minimum_image
    reference_point = System.reference_point
    formula = System.formula
    type_counts = System.type_counts

    def __init__(self, parent, members, placements, shell_types):
        self.parent = parent
        self.symbols = parent.symbols
        self.coordinates = parent.coordinates
        self.cell = parent.cell
        self.atomic_numbers = parent.atomic_numbers
        self._masses = parent.masses
        self.bonds = parent.bonds
        self.catalog = parent.catalog
        self.members = [tuple(m) for m in members]
        self.types = {}
        self.molecules = []
        shells = set()
        for index, (molecules, images) in enumerate(zip(members, placements)):
            if len(molecules) == 1:
                original = parent.molecules[molecules[0]]
                self.molecules.append(
                    Molecule(
                        index=index,
                        atoms=original.atoms,
                        type=original.type,
                        coordinates=original.coordinates,
                        designated=original.designated,
                    )
                )
                self.types[original.type] = parent.types[original.type]
                continue
            shells.add(index)
            atoms, xyz = [], []
            for m, image in zip(molecules, images):
                atoms.append(parent.molecules[m].atoms)
                xyz.append(parent.molecules[m].coordinates + parent.shift(image))
            atoms = np.concatenate(atoms)
            xyz = np.vstack(xyz)
            order = np.argsort(atoms)
            ion_atom = parent.molecules[molecules[0]].atoms[0]
            mtype = shell_types[index]
            self.types[mtype.name] = mtype
            self.molecules.append(
                Molecule(
                    index=index,
                    atoms=atoms[order],
                    type=mtype.name,
                    coordinates=xyz[order],
                    designated=int(np.nonzero(atoms[order] == ion_atom)[0][0]),
                )
            )
        self.shells = frozenset(shells)

    def shell_sizes(self):
        """{unit index: number of molecules besides the ion} for the shells."""
        return {u: len(self.members[u]) - 1 for u in sorted(self.shells)}


def ion_shells(system, rules=None):
    """The units of a system under ion-shell rules.

    Parameters
    ----------
    system : seamm_mbe.System
    rules : IonShellRules
        Default Li⁺ with O and F at 2.6 Å, nearest ion, at most 5 members.

    Returns
    -------
    UnitSystem
        Units numbered by their lowest molecule index, so a system without any
        shell keeps its molecule numbering.
    """
    rules = IonShellRules() if rules is None else rules
    ions = [
        m.index
        for m in system.molecules
        if len(m.atoms) == 1 and system.symbols[m.atoms[0]] in rules.cutoffs
    ]
    # Each candidate: (distance, ion, image of the molecule placing it at the ion)
    claims = {}
    for i in ions:
        ion_atom = system.molecules[i].atoms[0]
        ion_xyz = system.molecules[i].coordinates[0]
        partners = rules.cutoffs[system.symbols[ion_atom]]
        for molecule in system.molecules:
            if molecule.index in ions:
                continue
            best = None
            for k, a in enumerate(molecule.atoms):
                cutoff = partners.get(system.symbols[a])
                if cutoff is None:
                    continue
                # n places the ion nearest the atom, so -n places the
                # molecule nearest the ion
                n, vector = system.minimum_image(ion_xyz - molecule.coordinates[k])
                d = float(np.linalg.norm(vector))
                if d < cutoff and (best is None or d < best[0]):
                    best = (d, tuple(-v for v in n))
            if best is not None:
                claims.setdefault(molecule.index, []).append((best[0], i, best[1]))
    shells = {i: [] for i in ions}
    for m, candidates in claims.items():
        d, i, image = min(candidates, key=lambda c: (c[0], c[1]))
        shells[i].append((d, m, image))
    for i in ions:
        shells[i].sort()
        if rules.max_members is not None:
            shells[i] = shells[i][: int(rules.max_members)]
    in_shell = {m: i for i in ions for _, m, _ in shells[i]}

    members, placements, shell_types = [], [], {}
    units = []
    for molecule in system.molecules:
        m = molecule.index
        if m in in_shell:
            continue
        if m in shells and shells[m]:
            ordered = sorted(shells[m], key=lambda s: s[1])
            units.append(
                (
                    m,
                    (m, *(s[1] for s in ordered)),
                    ((0, 0, 0), *(s[2] for s in ordered)),
                )
            )
        else:
            units.append((m, (m,), ((0, 0, 0),)))
    units.sort(key=lambda u: min(u[1]))
    types = {}
    for index, (_, molecules, images) in enumerate(units):
        members.append(molecules)
        placements.append(images)
        if len(molecules) == 1:
            continue
        mtype = _shell_type(system, molecules, types)
        shell_types[index] = mtype
    return UnitSystem(system, members, placements, shell_types)


def _shell_type(system, molecules, types):
    """The MoleculeType of a shell: named by the ion and its members' types,
    e.g. "Li+[DMC3 PF6-]", with the summed charge."""
    ion = system.molecules[molecules[0]].type
    counts = {}
    for m in molecules[1:]:
        t = system.molecules[m].type
        counts[t] = counts.get(t, 0) + 1
    inside = " ".join(f"{t}{n if n > 1 else ''}" for t, n in sorted(counts.items()))
    name = f"{ion}[{inside}]"
    if name in types:
        return types[name]
    member_types = [system.types[system.molecules[m].type] for m in molecules]
    if any(t.multiplicity != 1 for t in member_types):
        raise ShellError(
            f"The shell {name} holds an open-shell molecule; shells of open-shell "
            "molecules are not supported."
        )
    symbols = [system.symbols[a] for m in molecules for a in system.molecules[m].atoms]
    mtype = MoleculeType(
        name=name,
        formula=hill_formula(symbols),
        signature="shell:" + name,
        charge=sum(t.charge for t in member_types),
        multiplicity=1,
    )
    types[name] = mtype
    return mtype
