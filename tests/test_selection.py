"""The selection rules and the cell-size bound of SelectionRules.check."""

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
    kwargs, bound = CASES[case]
    rules = seamm_mbe.SelectionRules(**kwargs)
    with pytest.raises(seamm_mbe.SelectionError, match="smallest width exceeds"):
        rules.check(argon(bound - 0.01))
    rules.check(argon(bound + 0.01))


def test_bound_uses_smallest_width():
    """A skewed cell is judged by its smallest perpendicular width, not its
    edge lengths."""
    rules = seamm_mbe.SelectionRules(max_order=2, cutoffs={2: 4.5})
    cell = [[9.5, 0, 0], [6.0, 8.9, 0], [0, 0, 20.0]]  # edges 9.5, 10.7, 20
    system = seamm_mbe.System(["Ar"], [[0, 0, 0]], cell)
    assert system.widths.min() < 9.0
    with pytest.raises(seamm_mbe.SelectionError):
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
    with pytest.raises(seamm_mbe.SelectionError, match="contact"):
        seamm_mbe.SelectionRules(
            max_order=2, criterion="contact", cutoffs={2: c + 0.02}
        ).check(system)


def test_clusters_have_no_bound():
    system = seamm_mbe.System(["Ar"], [[0, 0, 0]])
    seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 100, 3: 100}).check(system)


def test_the_enumerator_refuses_collisions_too():
    """Bypassing check(): three atoms 3 Å apart on a line in a 9 Å cell form a
    ring through the boundary, so {0, 1, 2} is a connected triple in three
    different ways. The enumeration refuses rather than dropping two."""
    system = argon(9.0, [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
    rules = seamm_mbe.SelectionRules(max_order=3, cutoffs={2: 3.5, 3: 3.5})
    with pytest.raises(seamm_mbe.SelectionError):
        rules.check(system)
    with pytest.raises(seamm_mbe.SelectionError, match="two different"):
        _Enumerator(system, rules).run()


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
