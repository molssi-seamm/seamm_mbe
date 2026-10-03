"""The increment algebra: completeness, pairwise potentials, the virial's
origin independence and the per-level ladders."""

import itertools

import numpy as np
import pytest

import seamm_mbe

from .helpers import water_system


def full_expansion(system, order):
    """Every subset of the molecules up to ``order``, all selected."""
    rules = seamm_mbe.SelectionRules(
        max_order=order,
        cutoffs={n: 1000.0 for n in range(2, order + 1)},
        rules={n: "compact" for n in range(3, order + 1)},
    )
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    seamm_mbe.assign_levels(fragments)  # all molecular
    return fragments


def argon_cluster(n, seed=3):
    rng = np.random.default_rng(seed)
    return seamm_mbe.System(["Ar"] * n, rng.normal(scale=3.0, size=(n, 3)))


def test_completeness_with_arbitrary_numbers():
    """With every subset included, the increments sum to delta of the whole
    system exactly, for any numbers at all (Möbius inversion) -- energy and
    every atom's force."""
    system = argon_cluster(5)
    fragments = full_expansion(system, 5)
    assert len(fragments) == 2**5 - 1
    rng = np.random.default_rng(11)
    high = {f.name: (rng.normal(), rng.normal(size=(f.n_atoms, 3))) for f in fragments}
    low = {f.name: (rng.normal(), rng.normal(size=(f.n_atoms, 3))) for f in fragments}
    correction = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    whole = fragments["n5_00_01_02_03_04"]
    assert correction.energy == pytest.approx(high[whole.name][0] - low[whole.name][0])
    expected = np.zeros((5, 3))
    expected[whole.atoms] = high[whole.name][1] - low[whole.name][1]
    assert np.allclose(correction.forces, expected)


def coulomb(xyz, charges):
    """Pairwise Coulomb-like energy (eV) and forces (eV/Å) between atoms."""
    energy = 0.0
    forces = np.zeros_like(xyz)
    for i, j in itertools.combinations(range(len(xyz)), 2):
        d = xyz[i] - xyz[j]
        r = np.linalg.norm(d)
        e = charges[i] * charges[j] / r
        energy += e
        forces[i] += e * d / r**2
        forces[j] -= e * d / r**2
    return energy, forces


def axilrod_teller(xyz, molecules, c=5.0):
    """A genuine 3-body energy between molecules' first atoms (no forces
    needed for the energy test)."""
    energy = 0.0
    for a, b, d in itertools.combinations(molecules, 3):
        i, j, k = xyz[a], xyz[b], xyz[d]
        energy += (
            c
            / (np.linalg.norm(i - j) * np.linalg.norm(j - k) * np.linalg.norm(i - k))
            ** 3
        )
    return energy


def test_pairwise_potential_has_no_higher_increments(pilot):
    """Waters with a pairwise atom-atom potential as [high - low]: the 3- and
    4-body increments vanish, energy and forces, which checks the slot
    bookkeeping of molecules with several atoms."""
    system = water_system(pilot.X[:15], None)  # five waters as a cluster
    fragments = full_expansion(system, 4)
    charges = np.array([-0.8, 0.4, 0.4] * 5)
    high, low = {}, {}
    for f in fragments:
        q = charges[f.atoms]
        high[f.name] = coulomb(f.coordinates, q)
        low[f.name] = coulomb(f.coordinates, 0.5 * q)
    correction = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    for name, increment in correction.increments.items():
        if increment.order >= 3:
            assert abs(increment.energy) < 1e-12, name
            assert np.abs(increment.forces).max() < 1e-12, name
    # and the sum is the full [high - low] (5-body truncated at 4 is exact here)
    e_high, f_high = coulomb(system.coordinates, charges)
    e_low, f_low = coulomb(system.coordinates, 0.5 * charges)
    assert correction.energy == pytest.approx(e_high - e_low, abs=1e-10)
    assert np.allclose(correction.forces, f_high - f_low, atol=1e-10)


def test_three_body_term_appears_only_in_triples():
    system = argon_cluster(5)
    fragments = full_expansion(system, 3)
    high, low = {}, {}
    for f in fragments:
        local = list(range(f.n_atoms))
        high[f.name] = (axilrod_teller(f.coordinates, local), np.zeros((f.n_atoms, 3)))
        low[f.name] = (0.0, np.zeros((f.n_atoms, 3)))
    correction = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    assert correction.per_body.get(2, {"energy": 0.0})["energy"] == pytest.approx(0.0)
    assert correction.per_body[3]["energy"] == pytest.approx(
        axilrod_teller(system.coordinates, list(range(5)))
    )


def test_virial_is_origin_independent():
    """Increments with a deliberately non-zero net force: the correction's
    virial does not move when the whole system is translated, although the
    naive sum r (x) f does."""
    rng = np.random.default_rng(5)
    results = None
    virials, naive = [], []
    for origin in ([0.0, 0, 0], [7.0, -3.0, 11.0]):
        system = seamm_mbe.System(
            ["Ar"] * 4, argon_cluster(4).coordinates + np.array(origin)
        )
        fragments = full_expansion(system, 3)
        if results is None:
            results = {
                f.name: (rng.normal(), rng.normal(size=(f.n_atoms, 3)) + 0.3)
                for f in fragments
            }
            zero = {f.name: (0.0, np.zeros((f.n_atoms, 3))) for f in fragments}
        correction = seamm_mbe.mbe_correction(fragments, results, molecular=zero)
        assert (
            max(np.abs(i.net_force).max() for i in correction.increments.values()) > 0.1
        )
        virials.append(correction.virial)
        naive.append(
            sum(
                np.einsum("ia,ib->ab", fragments[n].coordinates, i.forces)
                for n, i in correction.increments.items()
            )
        )
    assert np.allclose(virials[0], virials[1])
    assert not np.allclose(naive[0], naive[1])


def test_each_increment_uses_its_own_ladder(pilot, pilot_fragments):
    """The mixed scheme: a triple's increment subtracts the MOLECULAR-level
    increments of its pairs, even where the pair is periodic in the sum."""
    high, periodic, molecular = pilot.results()
    correction = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    name = next(
        t.name
        for t in pilot_fragments.by_order(3)
        if any(
            pilot_fragments[s].level == "periodic"
            for s, slots in t.subfragments
            if len(slots) == 2
        )
    )
    ladder = seamm_mbe.increments(
        pilot_fragments, pilot_fragments.calculations()["molecular"], high, molecular
    )
    assert correction.increments[name].level == "molecular"
    assert correction.increments[name].energy == pytest.approx(ladder[name][0])
    assert np.allclose(correction.increments[name].forces, ladder[name][1])
    pair = next(
        s
        for s, slots in pilot_fragments[name].subfragments
        if len(slots) == 2 and pilot_fragments[s].level == "periodic"
    )
    assert correction.increments[pair].level == "periodic"
    assert correction.increments[pair].energy != pytest.approx(ladder[pair][0])


def test_wrong_shapes_and_missing_levels(pilot_fragments, pilot):
    high, periodic, molecular = pilot.results()
    bad = dict(high)
    bad["m00"] = (bad["m00"][0], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="m00"):
        seamm_mbe.mbe_correction(pilot_fragments, bad, periodic, molecular)
    system = argon_cluster(3)
    fragments = seamm_mbe.enumerate_fragments(
        system, seamm_mbe.SelectionRules(max_order=2, cutoffs={2: 100.0})
    )
    with pytest.raises(ValueError, match="assign_levels"):
        seamm_mbe.mbe_correction(fragments, {})
    seamm_mbe.assign_levels(fragments)
    with pytest.raises(seamm_mbe.MissingFragmentsError) as error:
        seamm_mbe.mbe_correction(fragments, {})
    assert len(error.value.missing) == 2 * 6  # 3 monomers + 3 pairs, two levels


def test_levels_need_real_booleans():
    """Review item 6: an int 1 is a 1 Å distance, not True."""
    system = argon_cluster(4)
    fragments = full_expansion(system, 2)
    counts = seamm_mbe.assign_levels(fragments, {2: 1})
    assert counts["periodic"] == {}
    counts = seamm_mbe.assign_levels(fragments, {2: True})
    assert counts["periodic"] == {2: 6}
    with pytest.raises(ValueError, match="True, False or a distance"):
        seamm_mbe.assign_levels(fragments, {2: "yes"})


def test_increments_need_closed_names():
    system = argon_cluster(3)
    fragments = full_expansion(system, 2)
    zero = {f.name: (0.0, np.zeros((f.n_atoms, 3))) for f in fragments}
    with pytest.raises(ValueError, match="closed under sub-fragments"):
        seamm_mbe.increments(fragments, ["d00_01"], zero, zero)


def test_corrections_enter_only_the_sum():
    """A correction on a pair shifts the 2-body sum by exactly that amount and
    leaves every triple increment as it was (triples subtract the raw pair)."""
    system = argon_cluster(4)
    fragments = full_expansion(system, 3)
    rng = np.random.default_rng(21)
    high = {f.name: (rng.normal(), rng.normal(size=(f.n_atoms, 3))) for f in fragments}
    low = {f.name: (0.0, np.zeros((f.n_atoms, 3))) for f in fragments}
    plain = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    same = seamm_mbe.mbe_correction(fragments, high, molecular=low, corrections={})
    assert same.energy == plain.energy
    assert np.array_equal(same.forces, plain.forces)

    pair = fragments.by_order(2)[0]
    d_forces = rng.normal(size=(pair.n_atoms, 3))
    corrected = seamm_mbe.mbe_correction(
        fragments, high, molecular=low, corrections={pair.name: (0.25, d_forces)}
    )
    assert corrected.energy == pytest.approx(plain.energy + 0.25)
    assert corrected.per_body[2]["energy"] == pytest.approx(
        plain.per_body[2]["energy"] + 0.25
    )
    assert corrected.per_body[3]["energy"] == pytest.approx(plain.per_body[3]["energy"])
    for t in fragments.by_order(3):
        assert corrected.increments[t.name].energy == pytest.approx(
            plain.increments[t.name].energy
        )
    expected = plain.forces.copy()
    np.add.at(expected, pair.atoms, d_forces)
    assert np.allclose(corrected.forces, expected)
    # the correction is applied before the virial and the breakdown accumulate
    centred = d_forces - d_forces.mean(axis=0)
    d_virial = np.einsum("ia,ib->ab", pair.coordinates, centred)
    assert np.allclose(corrected.virial, plain.virial + d_virial)
    assert np.allclose(
        corrected.per_body[2]["virial"], plain.per_body[2]["virial"] + d_virial
    )
    assert np.allclose(corrected.per_body[3]["virial"], plain.per_body[3]["virial"])


def test_corrections_must_name_selected_fragments():
    system = argon_cluster(3)
    rules = seamm_mbe.SelectionRules(
        max_order=3, cutoffs={2: 3.0, 3: 100.0}, rules={3: "connected"}
    )
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    seamm_mbe.assign_levels(fragments)
    zero = {f.name: (0.0, np.zeros((f.n_atoms, 3))) for f in fragments}
    with pytest.raises(ValueError, match="not a selected fragment"):
        seamm_mbe.mbe_correction(
            fragments, zero, molecular=zero, corrections={"d07_09": (1.0, None)}
        )
    aux = fragments.by_order(2, in_sum=False)
    if aux:
        with pytest.raises(ValueError, match="not a selected fragment"):
            seamm_mbe.mbe_correction(
                fragments,
                zero,
                molecular=zero,
                corrections={aux[0].name: (1.0, np.zeros((6, 3)))},
            )
