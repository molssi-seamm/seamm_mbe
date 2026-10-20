"""A level of their own for the fragments holding an ion shell: e.g. the shell pairs
at a smaller basis than the solvent pairs, each such increment built entirely at
that level."""

import numpy as np
import pytest

import seamm_mbe
from tests.test_shells import pair_energy, shell_cluster


def three_body(xyz, charges):
    """A pair energy plus a small 3-body term, so the triples' increments are not
    zero."""
    energy, forces = pair_energy(xyz, charges)
    n = len(xyz)
    if n >= 3:
        centre = xyz.mean(axis=0)
        d = xyz - centre
        energy += 1e-3 * float((d**2).sum())
        forces -= 2e-3 * (d - d.mean(axis=0))
    return energy, forces


def fragments_with_shells(shell_max_order=2):
    system = shell_cluster()
    units = seamm_mbe.ion_shells(system)
    rules = seamm_mbe.SelectionRules(
        max_order=3,
        criterion="contact",
        cutoffs={2: 100.0, 3: 100.0},
        shell_max_order=shell_max_order,
    )
    fragments = seamm_mbe.enumerate_fragments(units, rules)
    seamm_mbe.assign_levels(fragments)
    q = np.random.default_rng(7).normal(size=len(system.symbols))
    high = {f.name: three_body(f.coordinates, q[f.atoms]) for f in fragments}
    low = {f.name: pair_energy(f.coordinates, 0.5 * q[f.atoms]) for f in fragments}
    return fragments, high, low


def holds_shell(fragments, f):
    return bool(set(f.molecules) & fragments.shells)


def test_level_key_prefers_the_shell():
    fragments, _, _ = fragments_with_shells(shell_max_order=None)
    levels = {3: "high:3", seamm_mbe.SHELL: "high:shell"}
    for f in fragments:
        key = seamm_mbe.level_key(f, levels, "high", fragments.shells)
        if holds_shell(fragments, f):
            assert key == "high:shell"
        else:
            assert key == ("high:3" if f.order == 3 else "high")
    assert seamm_mbe.level_key(fragments.fragments[0], None, "high") == "high"


def test_equal_levels_reproduce_a_single_level():
    fragments, high, low = fragments_with_shells()
    one = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    two = seamm_mbe.mbe_correction(
        fragments,
        high,
        molecular=low,
        high_by_order={"shell": high},
        low_by_order={"shell": low},
    )
    assert two.energy == pytest.approx(one.energy, abs=1e-12)
    assert np.allclose(two.forces, one.forces, atol=1e-12)
    assert np.allclose(two.virial, one.virial, atol=1e-12)
    for name, inc in two.increments.items():
        f = fragments[name]
        if holds_shell(fragments, f):
            assert (inc.high_level, inc.low_level) == ("high:shell", "molecular:shell")
        else:
            assert (inc.high_level, inc.low_level) == ("high", "molecular")


def test_a_shift_moves_only_the_shell_increments():
    """Shifting every shell-level high energy by c moves each shell monomer's
    increment by +c and each shell pair's by c - 2c = -c; the others stay."""
    fragments, high, low = fragments_with_shells()
    c = 0.01
    shifted = {name: (e + c, f) for name, (e, f) in high.items()}
    base = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    test = seamm_mbe.mbe_correction(
        fragments, high, molecular=low, high_by_order={"shell": shifted}
    )
    n1 = sum(
        1 for f in fragments.selected() if f.order == 1 and holds_shell(fragments, f)
    )
    n2 = sum(
        1 for f in fragments.selected() if f.order == 2 and holds_shell(fragments, f)
    )
    assert n1 > 0 and n2 > 0
    assert test.energy - base.energy == pytest.approx(c * (n1 - n2), abs=1e-9)
    for name, inc in test.increments.items():
        f = fragments[name]
        delta = inc.energy - base.increments[name].energy
        if not holds_shell(fragments, f):
            assert delta == pytest.approx(0.0, abs=1e-12)
        else:
            assert delta == pytest.approx(c if f.order == 1 else -c, abs=1e-9)
    assert np.allclose(test.forces, base.forces, atol=1e-12)


def test_calculations_list_the_shell_ladder():
    fragments, _, _ = fragments_with_shells()
    plain = fragments.calculations()
    split = fragments.calculations(
        high_levels={"shell": "high:shell"}, low_levels={"shell": "molecular:shell"}
    )
    shell_sel = [f for f in fragments.selected() if holds_shell(fragments, f)]
    other_sel = [f for f in fragments.selected() if not holds_shell(fragments, f)]
    assert split["high:shell"] == fragments.closure(shell_sel)
    assert split["high"] == fragments.closure(other_sel)
    assert split["molecular:shell"] == fragments.closure(shell_sel)
    assert split["molecular"] == fragments.closure(other_sel)
    # the solvent monomers of the shell pairs are needed at both levels
    both = set(split["high"]) & set(split["high:shell"])
    assert both and all(fragments[n].order == 1 for n in both)
    assert set(split["high"]) | set(split["high:shell"]) == set(plain["high"])
    # nothing missing when every level has every result
    _, high, low = fragments_with_shells()
    assert (
        seamm_mbe.missing_results(
            fragments,
            high,
            molecular=low,
            high_by_order={"shell": high},
            low_by_order={"shell": low},
        )
        == []
    )
    gaps = seamm_mbe.missing_results(
        fragments,
        high,
        molecular=low,
        high_by_order={"shell": {n: high[n] for n in split["high"]}},
    )
    assert {level for _, level in gaps} == {"high:shell"}


def test_shell_and_order_levels_together():
    """With shell triples (no truncation) and triples at their own level, a
    triple holding a shell takes the shell level; with equal results the
    correction is unchanged."""
    fragments, high, low = fragments_with_shells(shell_max_order=None)
    one = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    two = seamm_mbe.mbe_correction(
        fragments, high, molecular=low, high_by_order={3: high, "shell": high}
    )
    assert two.energy == pytest.approx(one.energy, abs=1e-12)
    levels = {
        (f.order, holds_shell(fragments, f)): two.increments[f.name].high_level
        for f in fragments.selected()
    }
    assert levels[(3, True)] == "high:shell"
    assert levels[(3, False)] == "high:3"
    assert levels[(2, False)] == "high"


def test_a_shell_key_without_shells_changes_nothing(pilot, pilot_fragments):
    high, periodic, molecular = pilot.results()
    one = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    two = seamm_mbe.mbe_correction(
        pilot_fragments, high, periodic, molecular, high_by_order={"shell": high}
    )
    assert two.energy == one.energy
    assert np.array_equal(two.forces, one.forces)
    assert not pilot_fragments.shells
