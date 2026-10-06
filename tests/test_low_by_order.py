"""A molecular low level per order: e.g. the pairs' low level at a large basis,
the triples' (with their own sub-pairs and monomers) at a smaller one. An
order's increment is a difference within its own ladder, so its levels need only
be consistent within it."""

import numpy as np
import pytest

import seamm_mbe


def test_equal_levels_reproduce_a_single_low_level(pilot, pilot_fragments):
    """With the same results for the triples' low level, the correction is
    exactly today's."""
    high, periodic, molecular = pilot.results()
    one = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    two = seamm_mbe.mbe_correction(
        pilot_fragments, high, periodic, molecular, low_by_order={3: molecular}
    )
    assert two.energy == one.energy
    assert np.array_equal(two.forces, one.forces)
    assert np.array_equal(two.virial, one.virial)
    for order in one.per_body:
        assert two.per_body[order]["energy"] == one.per_body[order]["energy"]
    triples = [i for i in two.increments.values() if i.order == 3]
    assert triples and {i.low_level for i in triples} == {"molecular:3"}
    assert {i.low_level for i in two.increments.values() if i.order < 3} <= set(
        seamm_mbe.LEVELS
    )


def test_both_levels_per_order_together(pilot, pilot_fragments):
    """A high and a low level of their own for the triples, each equal to the
    plain ones, also reproduce today's correction."""
    high, periodic, molecular = pilot.results()
    one = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    two = seamm_mbe.mbe_correction(
        pilot_fragments,
        high,
        periodic,
        molecular,
        high_by_order={3: high},
        low_by_order={3: molecular},
    )
    assert two.energy == one.energy
    assert np.array_equal(two.forces, one.forces)


def test_a_constant_shift_moves_only_the_triples(pilot, pilot_fragments):
    """Shifting every triples-low-level energy by c moves each triple's
    increment by exactly -c, and nothing else."""
    high, periodic, molecular = pilot.results()
    c = 0.01  # eV
    shifted = {
        name: (seamm_mbe.algebra._unpack(r)[0] + c, seamm_mbe.algebra._unpack(r)[1])
        for name, r in molecular.items()
    }
    base = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    test = seamm_mbe.mbe_correction(
        pilot_fragments, high, periodic, molecular, low_by_order={3: shifted}
    )
    n_triples = base.per_body[3]["count"]
    assert n_triples > 0
    assert test.per_body[3]["energy"] - base.per_body[3]["energy"] == pytest.approx(
        -n_triples * c, abs=1e-9
    )
    for order in (1, 2):
        assert test.per_body[order]["energy"] == base.per_body[order]["energy"]
    assert np.allclose(test.forces, base.forces, atol=1e-12)


def test_calculations_list_each_low_levels_ladder(pilot_fragments):
    plain = pilot_fragments.calculations()
    split = pilot_fragments.calculations(low_levels={3: "molecular:3"})
    assert split["periodic"] == plain["periodic"]
    assert split["high"] == plain["high"]
    # molecular:3: the molecular triples with all their sub-pairs and monomers
    triples = [
        f for f in pilot_fragments.selected() if f.order == 3 and f.level == "molecular"
    ]
    assert triples
    assert set(split["molecular:3"]) == set(pilot_fragments.closure(triples))
    # molecular: no triples any more, only the molecular monomers and pairs
    assert all(pilot_fragments[n].order <= 2 for n in split["molecular"])
    assert set(split["molecular"]) | set(split["molecular:3"]) == set(
        plain["molecular"]
    )


def test_a_missing_triples_low_level_result_is_refused(pilot, pilot_fragments):
    high, periodic, molecular = pilot.results()
    triples_level = dict(molecular)
    triple = next(t.name for t in pilot_fragments.by_order(3) if t.in_sum)
    del triples_level[triple]
    with pytest.raises(seamm_mbe.MissingFragmentsError) as info:
        seamm_mbe.mbe_correction(
            pilot_fragments, high, periodic, molecular, low_by_order={3: triples_level}
        )
    assert (triple, "molecular:3") in info.value.missing
