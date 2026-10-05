"""The many-body increments and the MBE correction: energy, forces, virial.

For a fragment F (a set of molecules at definite images) computed in isolation
at the high level and at a low level,

    delta_F = E_high(F) - E_low(F)

and its increment is the F-body part, with every proper sub-fragment's
increment removed (Möbius inversion over the subsets):

    dE_F = delta_F - sum over proper subsets S of F of dE_S

i.e. dE_i = delta_i, dE_ij = delta_ij - dE_i - dE_j, and so on. Forces go the
same way, atom by atom through the slots. The correction is the sum of the
selected fragments' increments, each at its own low level.

Each increment's virial, W_F = sum_a r_a (x) (f_a - <f>_F), uses its forces
less their mean, which makes it independent of the origin (a true increment
has zero net force; the residual is numerical but would otherwise be
multiplied by an arbitrary origin), with r_a the fragment's own coordinates.

Units: energies eV, forces eV/Å, virials eV (see :mod:`seamm_mbe.units`).
"""

from dataclasses import dataclass, field

import numpy as np

from .levels import LEVELS


class MissingFragmentsError(ValueError):
    """Results are missing or unusable, so the correction cannot be made.

    Attributes
    ----------
    missing : [(str, str)]
        (fragment name, level) of each missing result.
    """

    def __init__(self, missing):
        self.missing = list(missing)
        sample = ", ".join(f"{name} ({level})" for name, level in self.missing[:5])
        more = "" if len(self.missing) <= 5 else f" and {len(self.missing) - 5} more"
        super().__init__(
            f"{len(self.missing)} fragment result(s) are missing: {sample}{more}. "
            "The configuration is incomplete and gives no labels."
        )


@dataclass
class FragmentResult:
    """One fragment calculation: energy (eV) and forces (eV/Å, the fragment's
    atom order)."""

    energy: float
    forces: np.ndarray


def _unpack(result):
    if isinstance(result, FragmentResult):
        return float(result.energy), np.asarray(result.forces, dtype=float)
    if isinstance(result, dict):
        return float(result["energy"]), np.asarray(result["forces"], dtype=float)
    energy, forces = result
    return float(energy), np.asarray(forces, dtype=float)


@dataclass
class Increment:
    """One fragment's many-body increment.

    Attributes
    ----------
    name : str
    order : int
    level : str
        The low level it is referenced to.
    energy : float
        eV.
    forces : numpy.ndarray
        (n_atoms, 3) eV/Å in the fragment's atom order.
    net_force : numpy.ndarray
        (3,) eV/Å, the sum of the forces: ideally zero.
    virial : numpy.ndarray
        (3, 3) eV, origin-independent (net force removed).
    """

    name: str
    order: int
    level: str
    energy: float
    forces: np.ndarray
    net_force: np.ndarray
    virial: np.ndarray
    high_level: str = "high"


@dataclass
class Correction:
    """The MBE correction of a system.

    Attributes
    ----------
    energy : float
        eV.
    forces : numpy.ndarray
        (n_atoms, 3) eV/Å, on the system's atoms.
    virial : numpy.ndarray
        (3, 3) eV.
    increments : {str: Increment}
        The selected fragments' increments, each at its own level.
    per_body : {int: {"energy": eV, "virial": (3, 3) eV, "count": int}}
        The sums by order.
    per_level : {str: {int: int}}
        How many increments of each order used each level.
    max_increment_net_force : float
        The largest component of any increment's net force (eV/Å).
    """

    energy: float
    forces: np.ndarray
    virial: np.ndarray
    increments: dict = field(default_factory=dict)
    per_body: dict = field(default_factory=dict)
    per_level: dict = field(default_factory=dict)
    max_increment_net_force: float = 0.0


def increments(fragments, names, high, low):
    """The increments of the named fragments from high- and low-level results.

    Parameters
    ----------
    fragments : seamm_mbe.FragmentSet
    names : [str]
        Fragments to compute, closed under sub-fragments, ascending in order
        (as :meth:`FragmentSet.calculations` gives them).
    high, low : {str: FragmentResult or (energy, forces) or dict}
        Results by fragment name: energy in eV, forces in eV/Å.

    Returns
    -------
    {str: (float, numpy.ndarray)}
        Each fragment's increment: energy (eV) and forces (eV/Å).
    """
    result = {}
    for name in names:
        fragment = fragments[name]
        e_high, f_high = _unpack(high[name])
        e_low, f_low = _unpack(low[name])
        shape = (fragment.n_atoms, 3)
        if f_high.shape != shape or f_low.shape != shape:
            raise ValueError(
                f"Fragment {name}: forces of shape {f_high.shape} (high) and "
                f"{f_low.shape} (low), expected {shape}"
            )
        energy = e_high - e_low
        forces = f_high - f_low
        for sub_name, slots in fragment.subfragments:
            sub = fragments[sub_name]
            if sub_name not in result:
                raise ValueError(
                    f"The increment of {name} needs that of {sub_name}, which is "
                    "not earlier in 'names': the names must be closed under "
                    "sub-fragments and in ascending order."
                )
            sub_energy, sub_forces = result[sub_name]
            energy -= sub_energy
            for k, slot in enumerate(slots):
                forces[fragment.slot_atoms[slot]] -= sub_forces[sub.slot_atoms[k]]
        result[name] = (energy, forces)
    return result


def _high_levels(high_by_order):
    """{order: level name} for the orders with a high level of their own."""
    return {order: f"high:{order}" for order in (high_by_order or {})}


def missing_results(fragments, high, periodic=None, molecular=None, high_by_order=None):
    """The (name, level) of every result the correction needs but lacks.
    Results that are None count as missing (a failed calculation).
    ``high_by_order`` as for :func:`mbe_correction`; its levels are named
    "high:<order>"."""
    levels = _high_levels(high_by_order)
    needed = fragments.calculations(high_levels=levels)
    given = {"high": high, "periodic": periodic or {}, "molecular": molecular or {}}
    for order, key in levels.items():
        given[key] = high_by_order[order]
    return [
        (name, level)
        for level in given
        for name in needed.get(level, [])
        if given[level].get(name) is None
    ]


def mbe_correction(
    fragments,
    high,
    periodic=None,
    molecular=None,
    corrections=None,
    high_by_order=None,
):
    """The MBE correction from the fragment results.

    Parameters
    ----------
    fragments : seamm_mbe.FragmentSet
        With levels assigned (:func:`seamm_mbe.assign_levels`).
    high : {str: result}
        High-level results by fragment name.
    periodic, molecular : {str: result}
        Low-level results by fragment name, for the fragments
        :meth:`FragmentSet.calculations` lists at each level. A result is a
        :class:`FragmentResult`, an (energy, forces) pair or a dict with
        "energy" and "forces": eV and eV/Å in the fragment's atom order (its
        molecules in slot order, each molecule's atoms ascending).
    high_by_order : {int: {str: result}} or None
        High-level results for orders that have a high level of their own, e.g.
        ``{3: tz_results}`` with ``high`` at a larger basis for the monomers and
        pairs. Each order's increments are built entirely from its own level's
        results (the triple, its pairs and its monomers at that level), so the
        orders listed need their sub-fragments at that level too
        (:meth:`FragmentSet.calculations` with ``high_levels``). With equal
        results this gives exactly what ``high`` alone gives.
    corrections : {str: result} or None
        Corrections, each a :class:`FragmentResult`, an (energy, forces) pair or a
        dict with "energy" and "forces", added to selected fragments' increments
        in the sum only,
        never to the sub-fragment increments that higher fragments subtract
        (eV and eV/Å, as the results). This is how a pairwise counterpoise
        correction enters: with dE_ij^CP - dE_ij for each pair, the sum is
        E(1) + sum dE_ij^CP + sum dE_ijk, the triples still subtracting the
        uncorrected pairs, so a pair's BSSE does not move into the 3-body
        terms.

    Returns
    -------
    Correction

    Raises
    ------
    MissingFragmentsError
        If any needed result is missing: an incomplete configuration never
        gives a correction.
    """
    corrections = corrections or {}
    for name in corrections:
        if name not in fragments or not fragments[name].in_sum:
            raise ValueError(
                f"A correction is given for {name}, which is not a selected fragment."
            )
    for f in fragments.selected():
        if f.level not in LEVELS:
            raise ValueError(
                f"Fragment {f.name} has no level; call assign_levels() first."
            )
    missing = missing_results(fragments, high, periodic, molecular, high_by_order)
    if missing:
        raise MissingFragmentsError(missing)
    high_levels = _high_levels(high_by_order)
    highs = {"high": high}
    for order, key in high_levels.items():
        highs[key] = high_by_order[order]
    lows = {"periodic": periodic or {}, "molecular": molecular or {}}
    # One ladder per (low level, high level): each the closure of the selected
    # fragments that use that pair of levels, ascending in order
    ladders = {}
    for level in LEVELS:
        for key in highs:
            names = fragments.closure(
                f
                for f in fragments.selected()
                if f.level == level and high_levels.get(f.order, "high") == key
            )
            if names:
                ladders[(level, key)] = increments(
                    fragments, names, highs[key], lows[level]
                )

    n_atoms = len(fragments.system.symbols)
    total_forces = np.zeros((n_atoms, 3))
    total_virial = np.zeros((3, 3))
    total_energy = 0.0
    result = Correction(energy=0.0, forces=total_forces, virial=total_virial)
    for f in fragments.selected():
        key = high_levels.get(f.order, "high")
        energy, forces = ladders[(f.level, key)][f.name]
        if f.name in corrections:
            d_energy, d_forces = _unpack(corrections[f.name])
            if d_forces.shape != forces.shape:
                raise ValueError(
                    f"The correction of {f.name} has forces of shape "
                    f"{d_forces.shape}, expected {forces.shape}"
                )
            energy = energy + d_energy
            forces = forces + d_forces
        net = forces.sum(axis=0)
        centred = forces - net / len(forces)
        virial = np.einsum("ia,ib->ab", f.coordinates, centred)
        result.increments[f.name] = Increment(
            name=f.name,
            order=f.order,
            level=f.level,
            energy=energy,
            forces=forces,
            net_force=net,
            virial=virial,
            high_level=key,
        )
        total_energy += energy
        total_virial += virial
        np.add.at(total_forces, f.atoms, forces)
        body = result.per_body.setdefault(
            f.order, {"energy": 0.0, "virial": np.zeros((3, 3)), "count": 0}
        )
        body["energy"] += energy
        body["virial"] = body["virial"] + virial
        body["count"] += 1
        counts = result.per_level.setdefault(f.level, {})
        counts[f.order] = counts.get(f.order, 0) + 1
        result.max_increment_net_force = max(
            result.max_increment_net_force, float(np.abs(net).max())
        )
    result.energy = total_energy
    return result
