"""Fragment enumeration: canonical keys and names under image changes, outer
pairs, auxiliary sub-fragments, clusters."""

import itertools

import numpy as np
import pytest

import seamm_mbe

from .helpers import water_system


def summary(fragments):
    return [(f.name, f.in_sum, f.level, f.order) for f in fragments]


def assert_same_up_to_lattice(a, b, box):
    """Each fragment's coordinates differ by one lattice translation."""
    for fa, fb in zip(a, b):
        shift = fb.coordinates - fa.coordinates
        assert np.allclose(shift, shift[0], atol=1e-9), fa.name
        assert np.allclose(shift[0] / box, np.round(shift[0] / box), atol=1e-9)


def enumerate_pilot(X, box):
    fragments = seamm_mbe.enumerate_fragments(water_system(X, box))
    seamm_mbe.assign_levels(fragments, {1: True, 2: 3.5})
    return fragments


def test_translating_by_a_lattice_vector(pilot, pilot_fragments):
    box = pilot.box
    moved = enumerate_pilot(pilot.X + np.array([box, -2 * box, 0.0]), box)
    assert summary(moved) == summary(pilot_fragments)
    assert [f.key for f in moved] == [f.key for f in pilot_fragments]
    for a, b in zip(pilot_fragments, moved):
        assert np.allclose(b.coordinates - a.coordinates, [box, -2 * box, 0.0])


def test_moving_single_molecules_by_lattice_vectors(pilot, pilot_fragments):
    """The names and selection do not depend on which image of each molecule
    the input holds; the keys' raw images do, consistently."""
    box = pilot.box
    rng = np.random.default_rng(7)
    X = pilot.X.copy()
    for m in range(64):
        X[3 * m : 3 * m + 3] += box * rng.integers(-1, 2, size=3)
    moved = enumerate_pilot(X, box)
    assert summary(moved) == summary(pilot_fragments)
    assert_same_up_to_lattice(pilot_fragments, moved, box)


def test_wrapping_atoms_into_the_cell(pilot, pilot_fragments):
    box = pilot.box
    wrapped = enumerate_pilot(pilot.X - box * np.floor(pilot.X / box), box)
    assert summary(wrapped) == summary(pilot_fragments)
    assert_same_up_to_lattice(pilot_fragments, wrapped, box)


def test_keys_are_canonical(pilot_fragments):
    for f in pilot_fragments:
        placement = dict(zip(f.molecules, f.images))
        assert seamm_mbe.canonical_key(placement) == f.key
        # any common shift of the frame gives the same key
        shifted = {m: (i[0] + 2, i[1] - 1, i[2]) for m, i in placement.items()}
        assert seamm_mbe.canonical_key(shifted) == f.key
    assert len({f.key for f in pilot_fragments}) == len(pilot_fragments)


def test_outer_pairs(pilot_fragments):
    """Every sub-pair of a triple exists; those beyond the pair cutoff are
    outer pairs (not in the 2-body sum), and some are longer than L/2."""
    outer = pilot_fragments.by_order(2, in_sum=False)
    assert len(outer) == 268
    assert all(f.max_distance >= 4.5 for f in outer)
    assert any(f.max_distance > 12.4297 / 2 for f in outer)
    for t in pilot_fragments.by_order(3):
        assert len(t.subfragments) == 6
        for name, slots in t.subfragments:
            sub = pilot_fragments[name]
            for k, slot in enumerate(slots):
                assert sub.molecules[k] == t.molecules[slot]
                # The sub-fragment is the same geometry, up to a translation
                d = (
                    t.coordinates[t.slot_atoms[slot]]
                    - sub.coordinates[sub.slot_atoms[k]]
                )
                assert np.allclose(d, d[0])


def test_non_minimum_image_pair_name():
    """A triple whose outer pair is not the minimum image: the outer pair's
    name carries the image, and the minimum-image pair is a different
    fragment."""
    box = 11.0
    # Molecule 1 is the hub; 0 and 2 are bonded to it on opposite sides, so
    # 0 and 2 are 6.8 Å apart through the hub but 4.2 Å apart through the
    # boundary.
    points = [(1.0, 1, 1), (4.4, 1, 1), (7.8, 1, 1)]
    system = seamm_mbe.System(["Ar"] * 3, points, np.eye(3) * box)
    rules = seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 4.5, 3: 3.5})
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    names = fragments.names
    assert "d00_02" in names  # minimum image, 4.2 Å, selected
    assert fragments["d00_02"].images == ((0, 0, 0), (-1, 0, 0))
    # Through the hub, 2 is at the same image as 0: not the minimum image
    outer = [n for n in names if n.startswith("d00_02_x")]
    assert outer == ["d00_02_x000"]
    assert fragments["d00_02"].in_sum and not fragments["d00_02_x000"].in_sum
    assert fragments["d00_02_x000"].max_distance == pytest.approx(6.8)
    triple = fragments["t00_01_02"]
    assert "d00_02_x000" in [n for n, _ in triple.subfragments]


def ring_of_argon(n, radius, box=None):
    angles = 2 * np.pi * np.arange(n) / n
    points = np.stack([radius * np.cos(angles), radius * np.sin(angles), 0 * angles], 1)
    points += 25.0
    cell = None if box is None else np.eye(3) * box
    return seamm_mbe.System(["Ar"] * n, points, cell)


def test_four_body_auxiliary_fragments():
    """Connected 4-body chains around a ring of 6: their non-connected
    sub-triples are auxiliary, with images in their names in a cell."""
    rules = seamm_mbe.SelectionRules(
        max_order=4,
        cutoffs={2: 3.2, 3: 3.2, 4: 3.2},
        rules={3: "connected", 4: "connected"},
    )
    system = ring_of_argon(6, 3.0, box=50.0)  # neighbours 3.0 Å apart
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    counts = fragments.counts()
    assert counts[1]["selected"] == 6
    assert counts[2]["selected"] == 6  # ring bonds
    assert counts[3]["selected"] == 6  # chains of 3
    assert counts[4]["selected"] == 6  # chains of 4
    # The 1-3 pairs (from the chains of 3) and the 1-4 pairs (from the chains
    # of 4) are auxiliary: 6 + 3
    assert counts[2]["auxiliary"] == 9
    # Each 4-chain a-b-c-d has two non-connected sub-triples, a-b-d and a-c-d:
    # the 12 triples {i, i+1, i+3} and {i, i+2, i+3}
    assert counts[3]["auxiliary"] == 12
    aux = fragments.by_order(3, in_sum=False)
    assert all("_x" in f.name for f in aux)
    for q in fragments.by_order(4):
        assert len(q.subfragments) == 14  # 4 + 6 + 4 proper subsets
    # In a cluster the same set has no image suffixes
    cluster = seamm_mbe.enumerate_fragments(ring_of_argon(6, 3.0), rules)
    assert [f.name.split("_x")[0] for f in fragments] == cluster.names


def test_rules_hub_and_compact():
    """Four atoms: a square of side 3 (diagonals 4.24). With a 3.5 Å cutoff
    the square is connected but has no hub and is not compact; with 4.5 Å it
    is compact."""
    square = [(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0)]
    system = seamm_mbe.System(["Ar"] * 4, square)

    def n4(rule, cutoff):
        rules = seamm_mbe.SelectionRules(
            max_order=4,
            cutoffs={2: 5.0, 3: 5.0, 4: cutoff},
            rules={3: "none", 4: rule},
        )
        return len(seamm_mbe.enumerate_fragments(system, rules).by_order(4, True))

    assert n4("connected", 3.5) == 1
    assert n4("hub", 3.5) == 0
    assert n4("compact", 3.5) == 0
    assert n4("hub", 4.5) == 1
    assert n4("compact", 4.5) == 1


def test_cluster_mode(pilot):
    """Four waters cut out as a cluster: no images, the same machinery."""
    system = water_system(pilot.X[:12], None)
    rules = seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 100.0, 3: 100.0})
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    assert fragments.counts() == {
        1: {"selected": 4, "auxiliary": 0},
        2: {"selected": 6, "auxiliary": 0},
        3: {"selected": 4, "auxiliary": 0},
    }
    assert all(i == (0, 0, 0) for f in fragments for i in f.images)
    for f in fragments:
        expected = np.vstack([system.molecules[m].coordinates for m in f.molecules])
        assert np.allclose(f.coordinates, expected)


def test_fragment_attributes_for_counterpoise(pilot_system, pilot_fragments):
    """Per-slot atoms: a sub-fragment's ghosts are the parent's other atoms."""
    t = pilot_fragments["t00_17_30"]
    assert t.n_atoms == 9 and t.charge == 0 and t.multiplicity == 1
    assert [list(s) for s in t.slot_atoms] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert list(t.atoms) == [0, 1, 2, 51, 52, 53, 90, 91, 92]
    assert t.symbols(pilot_system) == ["O", "H", "H"] * 3
    for slots in itertools.combinations(range(3), 2):
        own = np.concatenate([t.slot_atoms[s] for s in slots])
        ghosts = sorted(set(range(9)) - set(own))
        assert len(ghosts) == 3


def brute_force_pairs(system, distance, cutoff):
    """Pairs (i, j, image) within the cutoff, by checking every image in a
    5x5x5 block."""
    found = set()
    n = len(system.molecules)
    for i in range(n):
        for j in range(i + 1, n):
            for image in itertools.product(range(-2, 3), repeat=3):
                if distance(i, j, image) < cutoff:
                    found.add((i, j, image))
    return found


@pytest.mark.parametrize("criterion", ["contact", "heavy contact", "com", "cog"])
def test_other_criteria(pilot, criterion):
    """The selected pairs under each criterion are exactly those a brute-force
    search finds."""
    system = water_system(pilot.X, pilot.box)
    cutoff = {"contact": 2.4, "heavy contact": 3.0, "com": 3.4, "cog": 3.4}[criterion]
    rules = seamm_mbe.SelectionRules(
        max_order=2, criterion=criterion, cutoffs={2: cutoff}
    )
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    selected = {
        (f.molecules[0], f.molecules[1], f.images[1]) for f in fragments.by_order(2)
    }

    def distance(i, j, image):
        a = system.molecules[i]
        b = system.molecules[j]
        xb = b.coordinates + system.shift(image)
        if criterion in ("com", "cog"):
            pa = system.reference_point(a, criterion)
            pb = system.reference_point(b, criterion) + system.shift(image)
            return np.linalg.norm(pb - pa)
        xa = a.coordinates
        if criterion == "heavy contact":
            xa, xb = xa[:1], xb[:1]  # the O atoms
        return np.sqrt(((xa[:, None] - xb[None]) ** 2).sum(-1)).min()

    assert selected == brute_force_pairs(system, distance, cutoff)
    assert 50 < len(selected) < 400
    if criterion == "heavy contact":
        # For water the heavy-atom contact is the O-O distance
        designated = seamm_mbe.enumerate_fragments(
            system, seamm_mbe.SelectionRules(max_order=2, cutoffs={2: cutoff})
        )
        assert fragments.names == designated.names
