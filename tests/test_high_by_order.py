"""A high level per order: e.g. pairs (and the monomers in their increments) at a
large basis, triples (with their own sub-pairs and monomers) at a smaller one."""

import numpy as np
import pytest

import seamm_mbe


def test_equal_levels_reproduce_a_single_high_level(pilot, pilot_fragments):
    """With the same results for the triples' level, the correction is exactly
    today's: the 3-body ladder rebuilds the same sub-increments."""
    high, periodic, molecular = pilot.results()
    one = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    two = seamm_mbe.mbe_correction(
        pilot_fragments, high, periodic, molecular, high_by_order={3: high}
    )
    assert two.energy == one.energy
    assert np.array_equal(two.forces, one.forces)
    assert np.array_equal(two.virial, one.virial)
    for order in one.per_body:
        assert two.per_body[order]["energy"] == one.per_body[order]["energy"]
        assert np.array_equal(
            two.per_body[order]["virial"], one.per_body[order]["virial"]
        )
    assert {i.high_level for i in two.increments.values() if i.order == 3} == {"high:3"}
    assert {i.high_level for i in two.increments.values() if i.order < 3} == {"high"}


def test_a_constant_shift_moves_only_the_triples(pilot, pilot_fragments):
    """Shifting every triples-level energy by c leaves the monomer and pair
    increments alone; each triple's increment moves by exactly c (inclusion-
    exclusion of a constant over a triple's 7 subsets: 1 - 3 + 3 = 1)."""
    high, periodic, molecular = pilot.results()
    c = 0.01  # eV
    shifted = {name: (e + c, f) for name, (e, f) in high.items()}
    base = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    test = seamm_mbe.mbe_correction(
        pilot_fragments, high, periodic, molecular, high_by_order={3: shifted}
    )
    n_triples = base.per_body[3]["count"]
    assert n_triples > 0
    assert test.per_body[3]["energy"] - base.per_body[3]["energy"] == pytest.approx(
        n_triples * c, abs=1e-9
    )
    for order in (1, 2):
        assert test.per_body[order]["energy"] == base.per_body[order]["energy"]
    assert test.energy - base.energy == pytest.approx(n_triples * c, abs=1e-9)
    assert np.allclose(test.forces, base.forces, atol=1e-12)


def test_calculations_list_each_levels_ladder(pilot_fragments):
    plain = pilot_fragments.calculations()
    split = pilot_fragments.calculations(high_levels={3: "high:3"})
    # The low levels are unchanged
    assert split["periodic"] == plain["periodic"]
    assert split["molecular"] == plain["molecular"]
    # high: the monomers and pairs in the sum (and their sub-fragments)
    assert all(pilot_fragments[n].order <= 2 for n in split["high"])
    in_sum = {f.name for f in pilot_fragments.selected() if f.order <= 2}
    assert in_sum <= set(split["high"])
    # high:3: the triples with all their sub-pairs and monomers
    triples = [f for f in pilot_fragments.selected() if f.order == 3]
    expected = set(pilot_fragments.closure(triples))
    assert set(split["high:3"]) == expected
    # Together they cover what one high level needed; the overlap is computed twice
    assert set(split["high"]) | set(split["high:3"]) == set(plain["high"])
    twice = set(split["high"]) & set(split["high:3"])
    assert twice and all(pilot_fragments[n].order <= 2 for n in twice)


def test_a_missing_triples_level_result_is_refused(pilot, pilot_fragments):
    high, periodic, molecular = pilot.results()
    triples_level = dict(high)
    pair = next(
        s
        for t in pilot_fragments.by_order(3)
        if t.in_sum
        for s, slots in t.subfragments
        if len(slots) == 2
    )
    del triples_level[pair]
    with pytest.raises(seamm_mbe.MissingFragmentsError) as info:
        seamm_mbe.mbe_correction(
            pilot_fragments, high, periodic, molecular, high_by_order={3: triples_level}
        )
    assert (pair, "high:3") in info.value.missing
