"""The selection rules and the cell-size bound of SelectionRules.check."""

import itertools
import warnings

import numpy as np
import pytest

import seamm_mbe
from seamm_mbe.fragments import _Enumerator

from .helpers import water_system


def argon(cell_length, positions=((0.0, 0.0, 0.0),)):
    return seamm_mbe.System(["Ar"] * len(positions), positions, np.eye(3) * cell_length)


# (rules, the bound L_min must exceed)
CASES = {
    "pairs: 2c": (dict(max_order=2, cutoffs={2: 4.5}), 9.0),
    "connected triples: 3c": (
        dict(max_order=3, cutoffs={2: 3.0, 3: 3.5}, rules={3: "connected"}),
        10.5,
    ),
    "compact triples: 2c": (
        dict(max_order=3, cutoffs={2: 3.0, 3: 4.0}, rules={3: "compact"}),
        8.0,
    ),
    "hub 4-body: 3c": (
        dict(
            max_order=4, cutoffs={2: 3.0, 3: 3.0, 4: 3.2}, rules={3: "none", 4: "hub"}
        ),
        9.6,
    ),
    "connected 4-body: 5c": (
        dict(
            max_order=4,
            cutoffs={2: 3.0, 3: 3.0, 4: 3.2},
            rules={3: "none", 4: "connected"},
        ),
        16.0,
    ),
    "compact 4-body: 2c": (
        dict(
            max_order=4,
            cutoffs={2: 3.0, 3: 3.0, 4: 4.0},
            rules={3: "none", 4: "compact"},
        ),
        8.0,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_bound(case):
    """Below the bound the same molecules may form several fragments: a warning,
    since they are enumerated separately. Above it, nothing."""
    kwargs, bound = CASES[case]
    rules = seamm_mbe.SelectionRules(**kwargs)
    with pytest.warns(UserWarning, match="several different fragments"):
        rules.check(argon(bound - 0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rules.check(argon(bound + 0.01))


def test_a_molecule_within_reach_of_its_own_image_is_refused():
    """A cell narrower than the pair cutoff puts each molecule within reach of
    its own image: no fragment can hold that, so the enumeration refuses it."""
    rules = seamm_mbe.SelectionRules(max_order=2, cutoffs={2: 4.5})
    with pytest.raises(seamm_mbe.SelectionError, match="its own image"):
        seamm_mbe.enumerate_fragments(argon(4.4), rules)


def rods(cell):
    """Three 8 Å carbon rods along z in a cell 10 Å wide in x and y, two of
    them in contact and one in contact through a periodic image."""
    symbols, xyz, bonds = [], [], []
    # the second 2.5 Å from the first; the third 2.5 Å from the first through x
    for x0, y0, z0 in ((1.0, 1.0, 2.0), (3.5, 1.0, 3.0), (9.6, 1.5, 12.0)):
        start = len(symbols)
        for k in range(6):
            symbols.append("C")
            xyz.append((x0, y0, z0 + 1.6 * k))
        bonds += [(start + k, start + k + 1) for k in range(5)]
    return seamm_mbe.System(symbols, xyz, cell, bonds=bonds)


def test_the_search_reaches_beyond_the_nearest_cells():
    """Long molecules (contact radius 4 Å) with a 3 Å contact cutoff in a cell
    10 Å wide: the reach, 3 + 2 x 4 = 11 Å, exceeds the cell, which the 27-cell
    search refused. Every partner within the cutoff is found, as a brute-force
    search over three image layers finds them."""
    from seamm_mbe.fragments import _Enumerator

    system = rods(np.diag([10.0, 10.0, 30.0]))
    rules = seamm_mbe.SelectionRules(max_order=2, criterion="contact", cutoffs={2: 3.0})
    enumerator = _Enumerator(system, rules)
    assert 3.0 + 2 * enumerator.radius > system.widths.min()
    found = [sorted(neighbors) for neighbors in enumerator.neighbors(2)]
    expected = [[] for _ in system.molecules]
    layers = range(-3, 4)
    for a in range(len(system.molecules)):
        for b in range(len(system.molecules)):
            if a == b:
                continue
            for image in itertools.product(layers, layers, layers):
                if enumerator.distance(a, (0, 0, 0), b, image) < 3.0:
                    expected[a].append((b, image))
    assert found == [sorted(e) for e in expected]
    assert any(found)
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    assert len(fragments.by_order(2)) == sum(len(e) for e in expected) // 2


def test_bound_uses_smallest_width():
    """A skewed cell is judged by its smallest perpendicular width, not its
    edge lengths."""
    rules = seamm_mbe.SelectionRules(max_order=2, cutoffs={2: 4.5})
    cell = [[9.5, 0, 0], [6.0, 8.9, 0], [0, 0, 20.0]]  # edges 9.5, 10.7, 20
    system = seamm_mbe.System(["Ar"], [[0, 0, 0]], cell)
    assert system.widths.min() < 9.0
    with pytest.warns(UserWarning, match="several different fragments"):
        rules.check(system)


def test_contact_criterion_pads_the_bound(pilot):
    """With contact distances the bound grows by 2 r_max per bond."""
    box = pilot.box
    system = water_system(pilot.X, box)
    radius = max(
        np.linalg.norm(m.coordinates - m.coordinates.mean(0), axis=1).max()
        for m in system.molecules
    )
    c = (box / 2 - 2 * radius) - 0.01
    seamm_mbe.SelectionRules(max_order=2, criterion="contact", cutoffs={2: c}).check(
        system
    )
    with pytest.warns(UserWarning, match="contact"):
        seamm_mbe.SelectionRules(
            max_order=2, criterion="contact", cutoffs={2: c + 0.02}
        ).check(system)


def test_clusters_have_no_bound():
    system = seamm_mbe.System(["Ar"], [[0, 0, 0]])
    seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 100, 3: 100}).check(system)


def test_the_same_molecules_through_different_images():
    """Three atoms 3 Å apart on a line in a 9 Å cell form a ring through the
    boundary: {0, 1, 2} makes three different connected triples, each with a
    different atom in the middle. All three are kept, with distinct names, the
    minimum-image one unsuffixed."""
    system = argon(9.0, [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
    rules = seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 3.5, 3: 3.5})
    with pytest.warns(UserWarning):
        rules.check(system)
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    triples = fragments.by_order(3, in_sum=True)
    assert len(triples) == 3
    assert all(t.molecules == (0, 1, 2) for t in triples)
    names = [t.name for t in triples]
    assert len(set(names)) == 3
    assert sum(1 for n in names if "_x" not in n) == 1
    # Each is a straight chain, 3 Å bonds, a different atom in the middle
    middles = set()
    for t in triples:
        x = np.sort(t.coordinates[:, 0])
        assert np.allclose(np.diff(x), 3.0)
        middle = int(np.argsort(t.coordinates[:, 0])[1])
        middles.add(t.molecules[middle])
    assert middles == {0, 1, 2}
    # Every sub-fragment an increment needs exists, at its images
    for t in triples:
        for sub, slots in t.subfragments:
            assert sub in fragments


def test_a_molecule_near_its_own_image_is_refused():
    system = argon(4.4)
    rules = seamm_mbe.SelectionRules(max_order=2, cutoffs={2: 4.5})
    with pytest.raises(seamm_mbe.SelectionError, match="own image"):
        _Enumerator(system, rules).run()


def test_a_hub_bonded_to_two_images_of_a_partner():
    """Atom 1 is 3 Å from atom 0 on both sides in a 6 Å cell, and atom 2 is
    bonded to atom 0 only. (1, 0, 1') would hold atom 1 twice and is skipped;
    (0, 1, 2) and (0, 1', 2) are different triples and both enumerated."""
    system = argon(6.0, [(0, 0, 0), (3, 0, 0), (0, 2, 0)])
    rules = seamm_mbe.SelectionRules(
        max_order=3, cutoffs={2: 3.5, 3: 3.5}, rules={3: "connected"}
    )
    with pytest.warns(UserWarning, match="several different fragments"):
        fragments = seamm_mbe.enumerate_fragments(system, rules)
    triples = list(fragments.by_order(3, in_sum=True))
    assert len(triples) == 2
    assert all(t.molecules == (0, 1, 2) for t in triples)
    assert {t.images[1] for t in triples} == {(0, 0, 0), (-1, 0, 0)}
    assert len({t.name for t in triples}) == 2
    pairs = {f.key for f in fragments.by_order(2, in_sum=True)}
    assert ((0, 1), ((0, 0, 0),)) in pairs
    assert ((0, 1), ((-1, 0, 0),)) in pairs
    # Every sub-fragment is there, the (1, 2) pairs at two different images
    for t in triples:
        for name, slots in t.subfragments:
            assert name in fragments
    outer = [f for f in fragments.by_order(2) if f.molecules == (1, 2)]
    assert len(outer) == 2 and not any(f.in_sum for f in outer)


def test_type_pair_tables():
    rules = seamm_mbe.SelectionRules(
        max_order=2,
        cutoffs={2: {("water", "water"): 4.5, ("Li+", "*"): 3.0, ("*", "*"): 5.0}},
    )
    assert rules.cutoff(2, "water", "water") == 4.5
    assert rules.cutoff(2, "water", "Li+") == 3.0
    assert rules.cutoff(2, "Li+", "EC") == 3.0
    assert rules.cutoff(2, "EC", "EC") == 5.0
    rules = seamm_mbe.SelectionRules(
        max_order=2, cutoffs={2: {("water", "water"): 4.5}}
    )
    with pytest.raises(seamm_mbe.SelectionError, match=r"\(water, EC\)"):
        rules.cutoff(2, "water", "EC")


def test_bad_rules():
    with pytest.raises(seamm_mbe.SelectionError, match="criterion"):
        seamm_mbe.SelectionRules(criterion="nearest")
    with pytest.raises(seamm_mbe.SelectionError, match="rule for order 3"):
        seamm_mbe.SelectionRules(rules={3: "chain"})
    with pytest.raises(seamm_mbe.SelectionError, match="No cutoff given for order 4"):
        seamm_mbe.SelectionRules(max_order=4, rules={3: "connected", 4: "hub"})
    with pytest.raises(seamm_mbe.SelectionError, match="positive"):
        seamm_mbe.SelectionRules(cutoffs={2: 0.0, 3: 3.5})


def test_rule_filtered_sets_do_not_collide():
    """Review item 1: two different CONNECTED placements of the same molecules
    are not a collision when at most one passes the rule. Four atoms 2.9 Å
    apart in a 9.1 Å cell close a ring through the boundary; check() allows
    the hub rule (3c = 9.0) and the enumeration must too."""
    system = argon(9.1, [(0, 0, 0), (2.9, 0, 0), (5.8, 0, 0), (8.7, 0, 0)])
    rules = seamm_mbe.SelectionRules(
        max_order=4, cutoffs={2: 3.0, 3: 3.0, 4: 3.0}, rules={3: "none", 4: "hub"}
    )
    rules.check(system)
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    assert fragments.counts()[2]["selected"] == 4
    # the same with compact triples in a 6.05 Å cell (2c = 6.0)
    rng = np.random.default_rng(0)
    rules = seamm_mbe.SelectionRules(
        max_order=3, cutoffs={2: 3.0, 3: 3.0}, rules={3: "compact"}
    )
    for _ in range(50):
        system = argon(6.05, rng.uniform(0, 6.05, size=(4, 3)))
        rules.check(system)
        seamm_mbe.enumerate_fragments(system, rules)


def test_empty_periodic_system():
    system = seamm_mbe.System([], np.zeros((0, 3)), np.eye(3) * 10.0)
    seamm_mbe.SelectionRules().check(system)


def test_ambiguous_tables_are_refused():
    with pytest.raises(seamm_mbe.SelectionError, match="both 3.0 and 5.0"):
        seamm_mbe.SelectionRules(
            max_order=2,
            cutoffs={2: {("water", "Li+"): 3.0, ("Li+", "water"): 5.0}},
        )
    rules = seamm_mbe.SelectionRules(
        max_order=2, cutoffs={2: {("water", "*"): 3.0, ("*", "Li+"): 5.0}}
    )
    assert rules.cutoff(2, "water", "water") == 3.0
    with pytest.raises(seamm_mbe.SelectionError, match="ambiguous"):
        rules.cutoff(2, "water", "Li+")


def test_image_triples_have_no_increment_for_a_pair_potential():
    """With a pairwise-additive energy every triple increment vanishes, which
    needs each image-triple's sub-pairs at the right images."""
    system = argon(9.0, [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
    rules = seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 3.5, 3: 3.5})
    with pytest.warns(UserWarning):
        fragments = seamm_mbe.enumerate_fragments(system, rules)
    seamm_mbe.assign_levels(fragments)

    def energy(f):
        x = f.coordinates
        e, g = 0.0, np.zeros_like(x)
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                d = x[j] - x[i]
                r = np.linalg.norm(d)
                e += 1.0 / r**6
                de = -6.0 / r**7 * d / r
                g[i] -= de
                g[j] += de
        return e, -g  # (energy, forces)

    high = {f.name: energy(f) for f in fragments}
    low = {f.name: (0.0, np.zeros_like(f.coordinates)) for f in fragments}
    ladder = seamm_mbe.increments(
        fragments, fragments.calculations()["molecular"], high, low
    )
    for t in fragments.by_order(3, in_sum=True):
        energy_t, forces_t = ladder[t.name]
        assert abs(energy_t) < 1e-12
        assert np.allclose(forces_t, 0.0, atol=1e-12)


def test_too_many_image_layers_are_refused():
    """The search is capped at three layers of images: (2k + 1)^3 cells."""
    system = rods(np.diag([10.0, 10.0, 30.0]))
    rules = seamm_mbe.SelectionRules(max_order=2, criterion="contact", cutoffs={2: 3.0})
    rules.cutoffs[2] = 30.0  # a reach of 38 Å in a 10 Å cell: 5 layers
    with pytest.raises(seamm_mbe.SelectionError, match="layers of images"):
        seamm_mbe.enumerate_fragments(system, rules)
