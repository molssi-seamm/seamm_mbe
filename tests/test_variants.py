"""Variants of a level for single fragments: e.g. a shell pair whose AutoAux
calculation fails rebuilt from the pair and its monomers with explicit auxiliary
basis sets ("high:shell@aux"), the other increments untouched (MBE cell f04,
2026-10-23)."""

import numpy as np
import pytest

import seamm_mbe
from tests.test_shell_levels import at_shell_level, fragments_with_shells


def one_shell_pair():
    fragments, high, low = fragments_with_shells()
    pairs = [f for f in fragments.selected() if at_shell_level(fragments, f)]
    assert pairs
    return fragments, high, low, pairs[0]


def test_equal_variant_results_change_nothing():
    fragments, high, low, pair = one_shell_pair()
    base = seamm_mbe.mbe_correction(
        fragments, high, molecular=low, high_by_order={"shell": high}
    )
    variants = {"high:shell": ("high:shell@aux", {pair.name})}
    test = seamm_mbe.mbe_correction(
        fragments,
        high,
        molecular=low,
        high_by_order={"shell": high},
        variants=variants,
        variant_results={"high:shell@aux": high},
    )
    assert test.energy == pytest.approx(base.energy, abs=1e-12)
    assert np.allclose(test.forces, base.forces, atol=1e-12)
    assert np.allclose(test.virial, base.virial, atol=1e-12)
    assert test.increments[pair.name].high_level == "high:shell@aux"
    others = [n for n in test.increments if n != pair.name]
    assert all(test.increments[n].high_level != "high:shell@aux" for n in others)


def test_the_variant_increment_comes_only_from_the_variant():
    """Shifting every variant energy by c moves the pair's increment by c - 2c and
    nothing else: its monomers' own 1-body increments stay at their level."""
    fragments, high, low, pair = one_shell_pair()
    c = 0.01
    shifted = {name: (e + c, f) for name, (e, f) in high.items()}
    kwargs = dict(molecular=low, high_by_order={"shell": high})
    base = seamm_mbe.mbe_correction(fragments, high, **kwargs)
    test = seamm_mbe.mbe_correction(
        fragments,
        high,
        variants={"high:shell": ("high:shell@aux", {pair.name})},
        variant_results={"high:shell@aux": shifted},
        **kwargs,
    )
    for name, inc in test.increments.items():
        delta = inc.energy - base.increments[name].energy
        expected = -c if name == pair.name else 0.0
        assert delta == pytest.approx(expected, abs=1e-9), name
    assert test.energy - base.energy == pytest.approx(-c, abs=1e-9)


def test_low_level_variant():
    fragments, high, low, pair = one_shell_pair()
    c = 0.02
    shifted = {name: (e + c, f) for name, (e, f) in low.items()}
    kwargs = dict(molecular=low, low_by_order={"shell": low})
    base = seamm_mbe.mbe_correction(fragments, high, **kwargs)
    test = seamm_mbe.mbe_correction(
        fragments,
        high,
        variants={"molecular:shell": ("molecular:shell@aux", {pair.name})},
        variant_results={"molecular:shell@aux": shifted},
        **kwargs,
    )
    inc = test.increments[pair.name]
    assert inc.low_level == "molecular:shell@aux"
    # the low level is subtracted: -(c - 2c) = +c
    assert inc.energy - base.increments[pair.name].energy == pytest.approx(c)


def test_calculations_move_the_fragment_to_its_variant():
    fragments, _, _, pair = one_shell_pair()
    levels = {"high_levels": {"shell": "high:shell"}}
    base = fragments.calculations(**levels)
    test = fragments.calculations(
        **levels, variants={"high:shell": ("high:shell@aux", {pair.name})}
    )
    assert test["high:shell@aux"] == fragments.closure([pair])
    # The pair itself no longer at the base level (its AutoAux task is not rerun);
    # the base level keeps what the other shell pairs need, its monomers among
    # them where those pairs share them
    others = [
        f
        for f in fragments.selected()
        if at_shell_level(fragments, f) and f.name != pair.name
    ]
    assert pair.name not in test["high:shell"]
    assert test["high:shell"] == fragments.closure(others)
    assert set(test["high:shell"]) <= set(base["high:shell"])
    for name, _ in pair.subfragments:
        assert name in test["high:shell@aux"]
    assert test["high"] == base["high"]


def test_missing_variant_results_are_missing():
    fragments, high, low, pair = one_shell_pair()
    with pytest.raises(seamm_mbe.MissingFragmentsError) as error:
        seamm_mbe.mbe_correction(
            fragments,
            high,
            molecular=low,
            high_by_order={"shell": high},
            variants={"high:shell": ("high:shell@aux", {pair.name})},
            variant_results={"high:shell@aux": {}},
        )
    assert {level for _, level in error.value.missing} == {"high:shell@aux"}
    assert {name for name, _ in error.value.missing} == set(fragments.closure([pair]))


def test_variants_are_checked():
    fragments, high, low, pair = one_shell_pair()
    unselected = [f for f in fragments if not f.in_sum]
    with pytest.raises(ValueError, match="not a selected fragment"):
        fragments.calculations(variants={"high": ("high@aux", {"nonsense"})})
    if unselected:
        with pytest.raises(ValueError, match="not a selected fragment"):
            fragments.calculations(
                variants={"high": ("high@aux", {unselected[0].name})}
            )
    with pytest.raises(ValueError, match="another name"):
        fragments.calculations(variants={"high": ("high", {pair.name})})
    with pytest.raises(ValueError, match="not a level"):
        seamm_mbe.mbe_correction(
            fragments,
            high,
            molecular=low,
            variants={"high:9": ("high:9@aux", {pair.name})},
            variant_results={"high:9@aux": high},
        )
