"""Assembling the labels of a configuration: the cell's own terms plus the MBE
correction, the energy reference, stress and pressures.

    E = sum of cell terms + dE_MBE (+ per-molecule-type energy offsets)
    F = sum of cell terms' forces + F_MBE
    W = sum of cell terms' virials + W_MBE

The cell terms are whatever the step computed on the whole cell: the periodic
low level, and its dispersion add-on (dftd4 r2SCAN-D4 for VASP) as a separate
term. This library never runs a calculation.

Units: eV, eV/Å, virial eV, stress eV/Å³ (sigma = -P), pressures atm.
"""

from dataclasses import dataclass, field

import numpy as np

from .units import EV_PER_A3_TO_ATM


@dataclass
class CellTerm:
    """A whole-cell contribution.

    Attributes
    ----------
    label : str
        e.g. "VASP r2SCAN", "D4".
    energy : float
        eV.
    forces : numpy.ndarray
        (n_atoms, 3) eV/Å, the system's atom order.
    virial : numpy.ndarray or None
        (3, 3) eV (W = V * P; see :mod:`seamm_mbe.units` for converting a
        code's stress); None for a cluster.
    """

    label: str
    energy: float
    forces: np.ndarray
    virial: np.ndarray | None = None


@dataclass
class Labels:
    """The labels of one configuration.

    Attributes
    ----------
    energy : float
        Total energy, eV, on the high level's absolute scale.
    reference_energy : float
        ``energy`` plus the per-type offsets (eV): e.g. the formation-energy
        scale of the training sets. Equal to ``energy`` without offsets.
    forces : numpy.ndarray
        (n_atoms, 3) eV/Å.
    virial : numpy.ndarray or None
        (3, 3) eV.
    stress : numpy.ndarray or None
        (3, 3) eV/Å³, sigma = -W/V (ASE/xnn).
    pressure : float or None
        Atomic configurational pressure tr(W)/3V, atm.
    molecular_pressure : float or None
        tr(W - W_intra)/3V, atm (W_intra about each molecule's centre of mass).
    breakdown : {str: {"energy": eV, "pressure": atm}}
        Per cell term, and "MBE" for the correction.
    per_body : {int: {"energy": eV, "pressure": atm, "count": int}}
        The correction by order.
    per_level : {str: {int: int}}
        Increments per low level and order.
    max_increment_net_force : float
        eV/Å, the largest component of any increment's net force.
    net_force : float
        eV/Å, the norm of the total net force.
    """

    energy: float
    reference_energy: float
    forces: np.ndarray
    virial: np.ndarray | None
    stress: np.ndarray | None
    pressure: float | None
    molecular_pressure: float | None
    breakdown: dict = field(default_factory=dict)
    per_body: dict = field(default_factory=dict)
    per_level: dict = field(default_factory=dict)
    max_increment_net_force: float = 0.0
    net_force: float = 0.0


def intramolecular_virial(system, forces):
    """W_intra = sum over molecules, sum over their atoms, (r_a - R_com) (x) f_a,
    with each molecule whole (eV, from forces in eV/Å). The molecular virial
    is W - W_intra; monomer increments drop out of it exactly."""
    forces = np.asarray(forces, dtype=float)
    W = np.zeros((3, 3))
    for molecule in system.molecules:
        xyz = molecule.coordinates
        m = system.masses[molecule.atoms]
        com = m @ xyz / m.sum()
        W += np.einsum("ia,ib->ab", xyz - com, forces[molecule.atoms])
    return W


def energy_offset(system, offsets):
    """The total offset (eV) for a system from per-type offsets {type: eV per
    molecule}; every type present needs one."""
    missing = sorted(set(system.type_counts()) - set(offsets))
    if missing:
        raise ValueError(f"No energy offset for the molecule type(s) {missing}")
    return sum(offsets[m.type] for m in system.molecules)


def _atm(virial, volume):
    return float(np.trace(virial) / 3 / volume * EV_PER_A3_TO_ATM)


def assemble(system, correction, cell_terms, offsets=None):
    """The labels of a configuration.

    Parameters
    ----------
    system : seamm_mbe.System
    correction : seamm_mbe.Correction
        From :func:`seamm_mbe.mbe_correction`.
    cell_terms : [CellTerm]
        The whole-cell calculations (eV, eV/Å, virial eV).
    offsets : {str: float} or None
        Energy offset per molecule, by type name (eV).

    Returns
    -------
    Labels
    """
    n_atoms = len(system.symbols)
    energy = correction.energy
    forces = correction.forces.copy()
    periodic = system.periodic
    virial = correction.virial.copy() if periodic else None
    volume = system.volume
    breakdown = {}
    for term in cell_terms:
        term_forces = np.asarray(term.forces, dtype=float)
        if term_forces.shape != (n_atoms, 3):
            raise ValueError(
                f"Cell term {term.label}: forces of shape {term_forces.shape}, "
                f"expected {(n_atoms, 3)}"
            )
        energy += term.energy
        forces += term_forces
        entry = {"energy": float(term.energy)}
        if periodic:
            if term.virial is None:
                raise ValueError(f"Cell term {term.label} has no virial")
            virial += np.asarray(term.virial, dtype=float)
            entry["pressure"] = _atm(term.virial, volume)
        breakdown[term.label] = entry
    breakdown["MBE"] = {"energy": correction.energy}
    per_body = {}
    for order, body in sorted(correction.per_body.items()):
        per_body[order] = {"energy": body["energy"], "count": body["count"]}
        if periodic:
            per_body[order]["pressure"] = _atm(body["virial"], volume)
    if periodic:
        breakdown["MBE"]["pressure"] = _atm(correction.virial, volume)
        stress = -virial / volume
        pressure = _atm(virial, volume)
        molecular = _atm(virial - intramolecular_virial(system, forces), volume)
    else:
        stress = pressure = molecular = None
    reference = energy + (energy_offset(system, offsets) if offsets else 0.0)
    return Labels(
        energy=float(energy),
        reference_energy=float(reference),
        forces=forces,
        virial=virial,
        stress=stress,
        pressure=pressure,
        molecular_pressure=molecular,
        breakdown=breakdown,
        per_body=per_body,
        per_level=correction.per_level,
        max_increment_net_force=correction.max_increment_net_force,
        net_force=float(np.linalg.norm(forces.sum(axis=0))),
    )
