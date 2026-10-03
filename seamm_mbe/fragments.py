"""Fragment enumeration with periodic images and canonical keys.

A fragment is a set of whole molecules, each at a definite periodic image.
Its **key** is canonical: the sorted molecule indices plus the integer images
of molecules 2..n relative to molecule 1, so the same physical fragment
reached two ways (a pair selected for itself and as part of a triple) has one
key and is computed once.

Its **name** is the prototype's (``gen_stage2_frame.py``), so results stored
under the prototype's names match:

* monomers ``m00``;
* pairs ``d00_17``, with ``_x`` and one character per cell axis (``m``, ``0``,
  ``p`` for -1, 0, +1) appended when the image of the second molecule is not
  its minimum image relative to the first, e.g. ``d03_41_x0p0`` (images
  beyond +-1, never needed so far, are written out: ``_x2_0_-1``);
* selected fragments of order 3 and 4 ``t00_17_30``, ``q00_17_30_41`` --
  without images, which is unique because :meth:`SelectionRules.check`
  guarantees the same molecules cannot form two selected fragments;
* auxiliary fragments of order >= 3 (sub-fragments of a selected fragment
  that are not selected themselves) carry the images of molecules 2..n
  relative to molecule 1 in the same ``_x`` form in a periodic system (in a
  cluster there are no images, and no suffix).

Every selected fragment's proper sub-fragments are computed too, because its
increment needs them (the prototype's "outer pair of a connected triple"):
they are in the set with ``in_sum = False`` unless selected themselves.

Each fragment has a **frame**: the images its molecules are placed at for the
calculation. Monomers are at home; pairs and auxiliary fragments have their
first molecule at home; selected fragments of order >= 3 have their hub (the
first molecule, in index order, bonded to the most others) at home, as the
prototype did. Frames differ only by lattice translations, which change
nothing physical.
"""

from dataclasses import dataclass, field
import itertools

import numpy as np

from .selection import SelectionError, SelectionRules

_PREFIX = {1: "m", 2: "d", 3: "t", 4: "q"}


def _image_text(image):
    if all(abs(v) <= 1 for v in image):
        return "".join("m" if v < 0 else ("p" if v > 0 else "0") for v in image)
    return "_".join(str(v) for v in image)


def _sub(a, b):
    return tuple(int(x - y) for x, y in zip(a, b))


def canonical_key(placement):
    """The canonical key of a fragment from its placement.

    Parameters
    ----------
    placement : {int: (int, int, int)}
        The image of each molecule, in any common frame.

    Returns
    -------
    tuple
        (molecule indices ascending, images of molecules 2..n relative to
        molecule 1).
    """
    molecules = tuple(sorted(placement))
    base = placement[molecules[0]]
    return molecules, tuple(_sub(placement[m], base) for m in molecules[1:])


@dataclass
class Fragment:
    """A fragment of whole molecules.

    Attributes
    ----------
    name : str
        The prototype-compatible name (see the module docstring), used as the
        task key.
    key : tuple
        The canonical key (see :func:`canonical_key`).
    molecules : tuple of int
        The molecules, ascending: the fragment's slot order.
    images : tuple of (int, int, int)
        The image of each molecule in the fragment's frame.
    in_sum : bool
        Whether its increment enters the energy (selected), or it is only a
        sub-fragment of selected ones.
    coordinates : numpy.ndarray
        (n_atoms, 3) Cartesian coordinates in Å in the frame; the atoms are the
        molecules' atoms, molecule by molecule in slot order.
    atoms : numpy.ndarray of int
        The system's atom index of each fragment atom.
    slot_atoms : [numpy.ndarray]
        Per slot, the positions (rows) of that molecule's atoms in the fragment
        -- with ``atoms``, what a counterpoise ghost set needs.
    distances : {(int, int): float}
        The criterion distance (Å) between each pair of slots, in the frame.
    charge : int
        Total charge.
    multiplicities : tuple of int
        Each molecule's multiplicity.
    subfragments : [(str, tuple of int)]
        Every proper sub-fragment (all non-empty proper subsets of the
        molecules) by name, with the slots of this fragment it occupies.
    level : str or None
        For a selected fragment: which low level its increment uses,
        "periodic" or "molecular" (see :func:`seamm_mbe.assign_levels`).
    """

    name: str
    key: tuple
    molecules: tuple
    images: tuple
    in_sum: bool
    coordinates: np.ndarray
    atoms: np.ndarray
    slot_atoms: list
    distances: dict
    charge: int
    multiplicities: tuple
    subfragments: list = field(default_factory=list)
    level: str | None = None

    @property
    def order(self):
        """The number of molecules."""
        return len(self.molecules)

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def multiplicity(self):
        """The fragment's multiplicity: a molecule's own, or 1 when every
        molecule is closed-shell. Open-shell molecules in a fragment of
        several are not supported (their coupling is ambiguous)."""
        if self.order == 1:
            return self.multiplicities[0]
        if any(m != 1 for m in self.multiplicities):
            raise NotImplementedError(
                f"Fragment {self.name} contains open-shell molecules; the "
                "multiplicity of such a fragment is not defined yet."
            )
        return 1

    @property
    def max_distance(self):
        """The largest criterion distance (Å) between two members (0 for a
        monomer)."""
        return max(self.distances.values(), default=0.0)

    def symbols(self, system):
        """The element symbols of the fragment's atoms."""
        return [system.symbols[a] for a in self.atoms]


class FragmentSet:
    """The fragments of a system, in a stable order: by order, then by key
    (the prototype's order for monomers, pairs and selected triples).

    Use :func:`enumerate_fragments` to make one.
    """

    def __init__(self, system, rules, fragments):
        self.system = system
        self.rules = rules
        self.fragments = fragments
        self._by_name = {f.name: f for f in fragments}
        if len(self._by_name) != len(fragments):
            raise SelectionError("Two fragments have the same name")

    def __len__(self):
        return len(self.fragments)

    def __iter__(self):
        return iter(self.fragments)

    def __getitem__(self, name):
        return self._by_name[name]

    def __contains__(self, name):
        return name in self._by_name

    @property
    def names(self):
        return [f.name for f in self.fragments]

    def by_order(self, order, in_sum=None):
        """The fragments of an order; only selected (True) or only auxiliary
        (False) ones if ``in_sum`` is given."""
        return [
            f
            for f in self.fragments
            if f.order == order and (in_sum is None or f.in_sum == in_sum)
        ]

    def selected(self):
        """The fragments whose increments enter the sum."""
        return [f for f in self.fragments if f.in_sum]

    def counts(self):
        """{order: {"selected": n, "auxiliary": n}}."""
        out = {}
        for f in self.fragments:
            entry = out.setdefault(f.order, {"selected": 0, "auxiliary": 0})
            entry["selected" if f.in_sum else "auxiliary"] += 1
        return out

    def closure(self, fragments):
        """The names of the given fragments and all their sub-fragments, in
        set order."""
        names = set()
        for f in fragments:
            names.add(f.name)
            names.update(name for name, _ in f.subfragments)
        return [f.name for f in self.fragments if f.name in names]

    def calculations(self, molecular_everywhere=False):
        """What must be computed, by level.

        Returns
        -------
        {str: [str]}
            Fragment names for "high" (every fragment any increment needs),
            "periodic" and "molecular" (the selected fragments assigned to
            that low level, with all their sub-fragments: each increment is
            built from sub-fragments at its own level).
            ``molecular_everywhere`` puts every fragment in "molecular", which
            costs more but allows a consistency check of the two low levels.
        """
        out = {}
        for level in ("periodic", "molecular"):
            out[level] = self.closure(f for f in self.selected() if f.level == level)
        if molecular_everywhere:
            out["molecular"] = self.names
        needed = set(out["periodic"]) | set(out["molecular"])
        out["high"] = [name for name in self.names if name in needed]
        return out


def enumerate_fragments(system, rules=None):
    """Enumerate the fragments of a system.

    Parameters
    ----------
    system : seamm_mbe.System
        The molecules, with or without a cell.
    rules : seamm_mbe.SelectionRules
        The selection; default the prototype's (pairs < 4.5 Å, connected
        triples < 3.5 Å, designated-atom distances).

    Returns
    -------
    FragmentSet

    Raises
    ------
    SelectionError
        If the selection is not well defined for the cell (see
        :meth:`SelectionRules.check`).
    """
    rules = SelectionRules() if rules is None else rules
    rules.check(system)
    return _Enumerator(system, rules).run()


class _Enumerator:
    def __init__(self, system, rules):
        self.system = system
        self.rules = rules
        self.molecules = system.molecules
        self.types = [m.type for m in self.molecules]
        criterion = rules.criterion
        self.points = np.array(
            [system.reference_point(m, criterion) for m in self.molecules]
        )
        if criterion in ("contact", "heavy contact"):
            self.contact_atoms = []
            for m in self.molecules:
                xyz = m.coordinates
                if criterion == "heavy contact":
                    heavy = [system.symbols[a] != "H" for a in m.atoms]
                    if any(heavy):
                        xyz = xyz[np.array(heavy)]
                self.contact_atoms.append(xyz)
            self.radius = max(
                float(np.linalg.norm(x - p, axis=1).max())
                for x, p in zip(self.contact_atoms, self.points)
            )
        else:
            self.contact_atoms = None
            self.radius = 0.0
        self._min_image = {}

    # ---------------------------------------------------------------- distances
    def distance(self, a, image_a, b, image_b):
        """The criterion distance (Å) between molecule a at image_a and
        molecule b at image_b."""
        shift = self.system.shift(_sub(image_b, image_a))
        if self.contact_atoms is None:
            return float(np.linalg.norm(self.points[b] + shift - self.points[a]))
        xa = self.contact_atoms[a]
        xb = self.contact_atoms[b] + shift
        return float(np.sqrt(((xa[:, None] - xb[None]) ** 2).sum(-1).min()))

    def minimum_image(self, a, b):
        """The image of b nearest to a (by reference points)."""
        if (a, b) not in self._min_image:
            image, _ = self.system.minimum_image(self.points[b] - self.points[a])
            self._min_image[(a, b)] = image
        return self._min_image[(a, b)]

    def neighbors(self, order):
        """Per molecule, the (molecule, image) within the order's cutoff."""
        system = self.system
        n = len(self.molecules)
        result = [[] for _ in range(n)]
        pad = 2 * self.radius
        if system.periodic:
            shifts = np.array(
                [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                dtype=float,
            )
        for a in range(n):
            vectors = self.points - self.points[a]
            if system.periodic:
                base = -np.round(np.linalg.solve(system.cell.T, vectors.T).T)
                images = base[:, None, :] + shifts[None]
                candidates = vectors[:, None, :] + images @ system.cell
            else:
                images = np.zeros((n, 1, 3))
                candidates = vectors[:, None, :]
            lengths = np.linalg.norm(candidates, axis=-1)
            for b in range(n):
                if b == a:
                    continue
                cutoff = self.rules.cutoff(order, self.types[a], self.types[b])
                for k in np.nonzero(lengths[b] < cutoff + pad)[0]:
                    image = tuple(int(v) for v in images[b, k])
                    if pad == 0.0 or self.distance(a, (0, 0, 0), b, image) < cutoff:
                        result[a].append((b, image))
        return result

    def bonded(self, order, a, image_a, b, image_b):
        cutoff = self.rules.cutoff(order, self.types[a], self.types[b])
        return self.distance(a, image_a, b, image_b) < cutoff

    # --------------------------------------------------------------- enumerate
    def run(self):
        selected = {}  # key -> placement, frame and order
        for m in range(len(self.molecules)):
            placement = {m: (0, 0, 0)}
            selected[canonical_key(placement)] = placement
        max_order = self.rules.max_order
        if max_order >= 2:
            for a, neighbors in enumerate(self.neighbors(2)):
                for b, image in neighbors:
                    if b > a:
                        placement = {a: (0, 0, 0), b: image}
                        selected[canonical_key(placement)] = placement
        for order in range(3, max_order + 1):
            if self.rules.rule(order) == "none":
                continue
            for placement in self.connected_sets(order):
                if self.passes(order, placement):
                    frame = self.hub_frame(order, placement)
                    selected[canonical_key(frame)] = frame
        # Names of the selected fragments must be unique: the check guarantees
        # it, and this refuses rather than silently dropping one.
        fragments = {}
        for key, placement in selected.items():
            name = self.name(key, placement, aux=False)
            if name in fragments:
                raise SelectionError(
                    f"Two different fragments of molecules {key[0]} were selected "
                    "(the cell is too small for the cutoffs)."
                )
            fragments[name] = (key, placement, True)
        # The sub-fragments the increments need
        keys = {key for key, _, _ in fragments.values()}
        for key, placement in list(selected.items()):
            molecules = key[0]
            for size in range(2, len(molecules)):
                for subset in itertools.combinations(molecules, size):
                    sub = {m: placement[m] for m in subset}
                    subkey = canonical_key(sub)
                    if subkey in keys:
                        continue
                    first = subkey[0][0]
                    frame = {m: _sub(sub[m], sub[first]) for m in subset}
                    name = self.name(subkey, frame, aux=True)
                    fragments[name] = (subkey, frame, False)
                    keys.add(subkey)
        # By order, then key; selected fragments of order >= 3 before the
        # auxiliary ones. Pairs are not split: the prototype sorted outer pairs
        # in with the others.
        ordered = sorted(
            fragments.items(),
            key=lambda x: (
                len(x[1][0][0]),
                len(x[1][0][0]) >= 3 and not x[1][2],
                x[1][0],
            ),
        )
        if len({k for _, (k, _, _) in ordered}) != len(ordered):
            raise SelectionError("Two fragments have the same key")
        by_key = {key: name for name, (key, _, _) in ordered}
        result = [
            self.make(name, key, placement, in_sum, by_key)
            for name, (key, placement, in_sum) in ordered
        ]
        return FragmentSet(self.system, self.rules, result)

    def connected_sets(self, order):
        """All sets of ``order`` distinct molecules connected by bonds within
        the order's cutoff, as placements (one per set of molecules)."""
        neighbors = self.neighbors(order)
        found = {}
        level = {
            canonical_key({m: (0, 0, 0)}): {m: (0, 0, 0)}
            for m in range(len(self.molecules))
        }
        for size in range(2, order + 1):
            grown = {}
            for placement in level.values():
                for m, image in placement.items():
                    for b, relative in neighbors[m]:
                        if b in placement:
                            continue
                        new = dict(placement)
                        new[b] = tuple(i + r for i, r in zip(image, relative))
                        key = canonical_key(new)
                        if key not in grown:
                            grown[key] = new
            level = grown
        for key, placement in level.items():
            molecules = key[0]
            if molecules in found and found[molecules][0] != key:
                raise SelectionError(
                    f"Molecules {molecules} form two different connected "
                    f"order-{order} fragments (the cell is too small for the "
                    "cutoffs)."
                )
            found.setdefault(molecules, (key, placement))
        return [placement for _, placement in found.values()]

    def degrees(self, order, placement):
        molecules = sorted(placement)
        degree = {m: 0 for m in molecules}
        for a, b in itertools.combinations(molecules, 2):
            if self.bonded(order, a, placement[a], b, placement[b]):
                degree[a] += 1
                degree[b] += 1
        return degree

    def passes(self, order, placement):
        rule = self.rules.rule(order)
        degree = self.degrees(order, placement)
        if rule == "connected":
            return True
        if rule == "hub":
            return max(degree.values()) == order - 1
        if rule == "compact":
            return min(degree.values()) == order - 1
        return False

    def hub_frame(self, order, placement):
        """The placement with the hub at home: the first molecule, in index
        order, bonded to the most others (the prototype's hub for a connected
        triple)."""
        degree = self.degrees(order, placement)
        best = max(degree.values())
        hub = next(m for m in sorted(degree) if degree[m] == best)
        return {m: _sub(placement[m], placement[hub]) for m in placement}

    # -------------------------------------------------------------------- names
    def name(self, key, placement, aux):
        molecules, images = key
        order = len(molecules)
        prefix = _PREFIX.get(order, f"n{order}_")
        name = prefix + "_".join(f"{m:02d}" for m in molecules)
        if order == 2:
            if images[0] != self.minimum_image(*molecules):
                name += "_x" + _image_text(images[0])
        elif order >= 3 and aux and self.system.periodic:
            name += "_x" + "".join(_image_text(i) for i in images)
        return name

    # ---------------------------------------------------------------- fragments
    def make(self, name, key, placement, in_sum, by_key):
        system = self.system
        molecules = key[0]
        images = tuple(tuple(placement[m]) for m in molecules)
        xyz, atoms, slot_atoms = [], [], []
        start = 0
        for m, image in zip(molecules, images):
            molecule = self.molecules[m]
            xyz.append(molecule.coordinates + system.shift(image))
            atoms.append(molecule.atoms)
            slot_atoms.append(np.arange(start, start + len(molecule.atoms)))
            start += len(molecule.atoms)
        distances = {
            (i, j): self.distance(molecules[i], images[i], molecules[j], images[j])
            for i, j in itertools.combinations(range(len(molecules)), 2)
        }
        subfragments = []
        for size in range(1, len(molecules)):
            for slots in itertools.combinations(range(len(molecules)), size):
                sub = {molecules[s]: placement[molecules[s]] for s in slots}
                subfragments.append((by_key[canonical_key(sub)], slots))
        types = [system.types[self.molecules[m].type] for m in molecules]
        return Fragment(
            name=name,
            key=key,
            molecules=molecules,
            images=images,
            in_sum=in_sum,
            coordinates=np.vstack(xyz),
            atoms=np.concatenate(atoms),
            slot_atoms=slot_atoms,
            distances=distances,
            charge=sum(t.charge for t in types),
            multiplicities=tuple(t.multiplicity for t in types),
            subfragments=subfragments,
        )
