"""Units and sign conventions.

The library works throughout in

* energy: **eV**
* length: **Å**
* forces: **eV/Å** (forces, not gradients: F = -dE/dr)
* virial: **eV**, the tensor W = sum_a r_a (x) f_a, so that for a
  configurational (no kinetic) pressure tensor P = W / V
* pressure tensor P: **eV/Å³**, positive = the system pushes outward
* stress sigma: **eV/Å³** in the ASE/xnn convention sigma = (1/V) dE/d(strain)
  = -P

SEAMM's Model Chemistry batch contract (``analyze_task``) and the step's
``store_results`` use kJ/mol, kJ/mol/Å *gradients* and GPa; the functions here
convert both ways. The conversion layer is where unit bugs live, so every
function states its units and the tests check them against pint.

Codes disagree on the sign of the "stress" they print, so nothing here guesses:
every stress or pressure from outside comes with an explicit ``convention``,
either ``"pressure"`` (positive = pushes outward: VASP's ``in kB`` line, MDI's
``<STRESS``) or ``"stress"`` (sigma = -P: ASE, xnn).
"""

import numpy as np

#: 1 eV in kJ/mol (CODATA 2018: e * N_A / 1000, exact)
EV_TO_KJ_PER_MOL = 1.602176634e-19 * 6.02214076e23 / 1000.0
#: 1 eV/Å³ in GPa (exact: 1.602176634e-19 J / 1e-30 m³ = 1.602176634e11 Pa)
EV_PER_A3_TO_GPA = 160.2176634
#: 1 eV/Å³ in atm (1 atm = 101325 Pa)
EV_PER_A3_TO_ATM = 1.602176634e11 / 101325.0
#: 1 kbar (VASP's "kB") in eV/Å³
KBAR_TO_EV_PER_A3 = 0.1 / EV_PER_A3_TO_GPA

CONVENTIONS = ("pressure", "stress")


def _sign(convention):
    """+1 for a pressure (positive outward), -1 for a stress (sigma = -P)."""
    if convention == "pressure":
        return 1.0
    if convention == "stress":
        return -1.0
    raise ValueError(
        f"Unknown stress convention {convention!r}: it must be one of {CONVENTIONS}"
        " -- 'pressure' if positive means pushing outward (VASP, MDI), 'stress' "
        "if sigma = -P (ASE, xnn)."
    )


def as_tensor(value):
    """A (3, 3) array from a (3, 3) array, 9 values (row-major) or Voigt 6
    values in the order xx yy zz yz xz xy. Units are unchanged."""
    value = np.asarray(value, dtype=float)
    if value.shape == (3, 3):
        return value.copy()
    if value.size == 9:
        return value.reshape(3, 3)
    if value.size == 6:
        xx, yy, zz, yz, xz, xy = value
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    raise ValueError(f"A tensor needs 9 or 6 values, not shape {value.shape}")


def kj_per_mol_to_ev(energy):
    """Energy in kJ/mol -> eV."""
    return np.asarray(energy, dtype=float) / EV_TO_KJ_PER_MOL


def ev_to_kj_per_mol(energy):
    """Energy in eV -> kJ/mol."""
    return np.asarray(energy, dtype=float) * EV_TO_KJ_PER_MOL


def gradients_to_forces(gradients):
    """Gradients in kJ/mol/Å -> forces in eV/Å (note the sign: F = -dE/dr)."""
    return -np.asarray(gradients, dtype=float) / EV_TO_KJ_PER_MOL


def forces_to_gradients(forces):
    """Forces in eV/Å -> gradients in kJ/mol/Å (note the sign: g = -F)."""
    return -np.asarray(forces, dtype=float) * EV_TO_KJ_PER_MOL


def virial_from_tensor(value, volume, *, convention, units="GPa"):
    """The virial W (eV) from a stress or pressure tensor.

    Parameters
    ----------
    value : array-like
        (3, 3), 9 or Voigt 6 values (xx yy zz yz xz xy).
    volume : float
        The cell volume in Å³.
    convention : str
        "pressure" (positive = pushes outward; VASP, MDI) or "stress"
        (sigma = -P; ASE, xnn). Required: there is no default.
    units : str
        "GPa", "kbar" (VASP's kB), "eV/Å^3" or "atm".

    Returns
    -------
    numpy.ndarray
        (3, 3) virial in eV, W = V * P.
    """
    factor = {
        "GPa": 1.0 / EV_PER_A3_TO_GPA,
        "kbar": KBAR_TO_EV_PER_A3,
        "kB": KBAR_TO_EV_PER_A3,
        "eV/Å^3": 1.0,
        "eV/A^3": 1.0,
        "atm": 1.0 / EV_PER_A3_TO_ATM,
    }
    if units not in factor:
        raise ValueError(f"Unknown pressure units {units!r}: one of {list(factor)}")
    return _sign(convention) * as_tensor(value) * factor[units] * volume


def tensor_from_virial(virial, volume, *, convention, units="GPa"):
    """A stress or pressure tensor from the virial W (eV): the inverse of
    :func:`virial_from_tensor`, with the same arguments. Returns (3, 3) in
    ``units``."""
    unit = virial_from_tensor(np.eye(3), 1.0, convention="pressure", units=units)[0, 0]
    return _sign(convention) * np.asarray(virial, dtype=float) / volume / unit


def vasp_stress_to_virial(stress_kB, volume):
    """The virial (eV) from VASP's ``in kB`` line: XX YY ZZ XY YZ ZX in kbar,
    positive = pushing outward (a pressure). Note VASP's order differs from
    Voigt's."""
    xx, yy, zz, xy, yz, zx = np.asarray(stress_kB, dtype=float)
    pressure = np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
    return virial_from_tensor(pressure, volume, convention="pressure", units="kbar")


def dftd4_virial_to_virial(dE_dstrain):
    """The virial (eV) from the dftd4 library's "virial", which is
    dE/d(strain) (convert it from hartree to eV first): W = -dE/d(strain)."""
    return -np.asarray(dE_dstrain, dtype=float)


def pressure(virial, volume, units="atm"):
    """The scalar pressure, tr(P)/3, from the virial (eV) and volume (Å³), in
    ``units`` ("atm", "GPa", "kbar" or "eV/Å^3")."""
    tensor = tensor_from_virial(virial, volume, convention="pressure", units=units)
    return float(np.trace(tensor) / 3.0)


def from_analyze_task(result, *, volume=None, stress_convention=None):
    """Convert a Model Chemistry ``analyze_task`` result to the library's units.

    Parameters
    ----------
    result : dict
        {"energy": kJ/mol, "gradients": (n, 3) kJ/mol/Å, and optionally
        "stress": GPa as the program gives it}.
    volume : float
        The cell volume in Å³; needed only with a stress.
    stress_convention : str
        "pressure" or "stress" (see :func:`virial_from_tensor`); needed only
        with a stress, because the contract does not fix the sign.

    Returns
    -------
    dict
        {"energy": eV, "forces": (n, 3) eV/Å, and "virial": (3, 3) eV if the
        result had a stress}.
    """
    out = {
        "energy": float(kj_per_mol_to_ev(result["energy"])),
        "forces": gradients_to_forces(result["gradients"]),
    }
    if result.get("stress") is not None:
        if volume is None or stress_convention is None:
            raise ValueError(
                "A result with a stress needs the cell volume and the program's "
                "stress_convention ('pressure' or 'stress') to convert it."
            )
        out["virial"] = virial_from_tensor(
            result["stress"], volume, convention=stress_convention, units="GPa"
        )
    return out


def to_seamm(energy=None, forces=None, virial=None, volume=None):
    """The library's numbers in SEAMM's ``store_results`` conventions.

    energy (eV) -> kJ/mol; forces (eV/Å) -> gradients (kJ/mol/Å); virial (eV)
    with the volume (Å³) -> stress in GPa, **sigma = -P** (the ASE/xnn stress,
    the training-label convention). Returns a dict with the keys given.
    """
    out = {}
    if energy is not None:
        out["energy"] = float(ev_to_kj_per_mol(energy))
    if forces is not None:
        out["gradients"] = forces_to_gradients(forces)
    if virial is not None:
        out["stress"] = tensor_from_virial(
            virial, volume, convention="stress", units="GPa"
        )
    return out
