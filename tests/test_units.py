"""Units and sign conventions, checked against pint (seamm_util's Q_), not
against hand-copied constants."""

import numpy as np
import pytest
from seamm_util import Q_

from seamm_mbe import units


def test_constants_against_pint():
    assert units.EV_TO_KJ_PER_MOL == pytest.approx(
        Q_(1.0, "eV").m_as("kJ/mol"), rel=1e-12
    )
    assert units.EV_PER_A3_TO_GPA == pytest.approx(
        Q_(1.0, "eV/Å^3").m_as("GPa"), rel=1e-12
    )
    assert units.EV_PER_A3_TO_ATM == pytest.approx(
        Q_(1.0, "eV/Å^3").m_as("atm"), rel=1e-12
    )
    assert units.KBAR_TO_EV_PER_A3 == pytest.approx(
        Q_(1.0, "kbar").m_as("eV/Å^3"), rel=1e-12
    )


def test_energy_round_trip():
    e = 123.456
    assert units.kj_per_mol_to_ev(e) == pytest.approx(
        Q_(e, "kJ/mol").m_as("eV"), rel=1e-12
    )
    assert units.ev_to_kj_per_mol(units.kj_per_mol_to_ev(e)) == pytest.approx(e)


def test_gradient_gives_force_of_opposite_sign():
    """A gradient of +1 kJ/mol/Å along x is a force of -1 kJ/mol/Å, in eV/Å."""
    gradients = np.array([[1.0, 0.0, -2.0]])
    forces = units.gradients_to_forces(gradients)
    expected = -Q_(gradients, "kJ/mol/Å").m_as("eV/Å")
    assert np.allclose(forces, expected, rtol=1e-12)
    assert forces[0, 0] < 0 and forces[0, 2] > 0
    assert np.allclose(units.forces_to_gradients(forces), gradients)


def test_pressure_and_stress_conventions():
    """1 GPa of pressure (pushing outward) on 10 Å³ is a positive virial; the
    same tensor read as a stress (sigma = -P) is a negative one."""
    volume = 10.0
    tensor = np.eye(3)
    w_p = units.virial_from_tensor(tensor, volume, convention="pressure")
    w_s = units.virial_from_tensor(tensor, volume, convention="stress")
    expected = Q_(1.0, "GPa").m_as("eV/Å^3") * volume
    assert np.allclose(w_p, expected * np.eye(3), rtol=1e-12)
    assert np.allclose(w_s, -w_p)
    assert units.pressure(w_p, volume, "GPa") == pytest.approx(1.0)
    for convention in units.CONVENTIONS:
        for unit in ("GPa", "kbar", "atm", "eV/Å^3"):
            w = units.virial_from_tensor(
                tensor, volume, convention=convention, units=unit
            )
            back = units.tensor_from_virial(
                w, volume, convention=convention, units=unit
            )
            assert np.allclose(back, tensor)


def test_convention_is_required():
    with pytest.raises(TypeError):
        units.virial_from_tensor(np.eye(3), 1.0)
    with pytest.raises(ValueError, match="stress convention"):
        units.virial_from_tensor(np.eye(3), 1.0, convention="virial")


def test_voigt_order():
    """Voigt 6 is xx yy zz yz xz xy."""
    tensor = units.as_tensor([1, 2, 3, 4, 5, 6])
    assert tensor[1, 2] == tensor[2, 1] == 4
    assert tensor[0, 2] == tensor[2, 0] == 5
    assert tensor[0, 1] == tensor[1, 0] == 6
    assert np.allclose(units.as_tensor(np.arange(9)), np.arange(9).reshape(3, 3))


def test_vasp_kB_order_and_sign():
    """VASP's 'in kB' is XX YY ZZ XY YZ ZX, positive = pushing outward."""
    volume = 1000.0
    w = units.vasp_stress_to_virial([10, 20, 30, 1, 2, 3], volume)
    factor = Q_(1.0, "kbar").m_as("eV/Å^3") * volume
    assert w[0, 0] == pytest.approx(10 * factor)
    assert w[0, 1] == pytest.approx(1 * factor)  # XY
    assert w[1, 2] == pytest.approx(2 * factor)  # YZ
    assert w[0, 2] == pytest.approx(3 * factor)  # ZX
    assert units.pressure(w, volume, "kbar") == pytest.approx(20.0)


def test_dftd4_virial_sign():
    """dftd4 returns dE/d(strain). Dispersion binds: expanding raises the
    energy, dE/d(strain) > 0, so the pressure is negative."""
    dE_dstrain = np.eye(3) * 0.5
    w = units.dftd4_virial_to_virial(dE_dstrain)
    assert units.pressure(w, 100.0) < 0


def test_from_analyze_task():
    volume = 50.0
    result = {
        "energy": -100.0,
        "gradients": [[1.0, 2.0, 3.0]],
        "stress": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
    }
    with pytest.raises(ValueError, match="stress_convention"):
        units.from_analyze_task(result, volume=volume)
    out = units.from_analyze_task(result, volume=volume, stress_convention="pressure")
    assert out["energy"] == pytest.approx(Q_(-100.0, "kJ/mol").m_as("eV"))
    assert np.allclose(
        out["forces"], -Q_(np.array([[1.0, 2.0, 3.0]]), "kJ/mol/Å").m_as("eV/Å")
    )
    assert units.pressure(out["virial"], volume, "GPa") == pytest.approx(1.0)
    no_stress = units.from_analyze_task({"energy": 1.0, "gradients": [[0, 0, 0]]})
    assert "virial" not in no_stress


def test_to_seamm_is_the_inverse_with_sigma():
    """Back to SEAMM's conventions: kJ/mol, kJ/mol/Å gradients, and the
    stress in GPa as sigma = -P (positive pressure -> negative diagonal)."""
    volume = 50.0
    virial = units.virial_from_tensor(np.eye(3) * 2.0, volume, convention="pressure")
    out = units.to_seamm(
        energy=1.0, forces=[[0.1, 0.0, 0.0]], virial=virial, volume=volume
    )
    assert out["energy"] == pytest.approx(Q_(1.0, "eV").m_as("kJ/mol"))
    assert out["gradients"][0, 0] == pytest.approx(-Q_(0.1, "eV/Å").m_as("kJ/mol/Å"))
    assert np.allclose(out["stress"], -2.0 * np.eye(3))
    back = units.from_analyze_task(
        {
            "energy": out["energy"],
            "gradients": out["gradients"],
            "stress": out["stress"],
        },
        volume=volume,
        stress_convention="stress",
    )
    assert back["energy"] == pytest.approx(1.0)
    assert np.allclose(back["forces"], [[0.1, 0.0, 0.0]])
    assert np.allclose(back["virial"], virial)
