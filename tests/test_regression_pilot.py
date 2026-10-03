"""The pilot-frame regression: seamm_mbe reproduces the prototype
(assemble_frame.py) from the stored fragment results. See tests/data/README.md.
"""

import numpy as np
import pytest

import seamm_mbe
from seamm_mbe.units import EV_TO_KJ_PER_MOL

from .helpers import DFE_PER_WATER, water_system

#: The prototype's kJ/mol per eV (its per-body energies use it)
PROTOTYPE_KJ = 96.485332
#: The prototype's kB -> atm factor; the exact one is 1e8 / 101325
PROTOTYPE_KB_ATM = 986.923
EXACT_KB_ATM = 1e8 / 101325


def test_counts(pilot, pilot_fragments):
    counts = pilot_fragments.counts()
    assert counts[1] == {"selected": 64, "auxiliary": 0}
    assert counts[2] == {"selected": 391, "auxiliary": 268}
    assert counts[3] == {"selected": 559, "auxiliary": 0}
    calculations = pilot_fragments.calculations()
    # 218 periodic fragments + the cell = the prototype's 219 VASP inputs
    assert len(calculations["periodic"]) + 1 == 219
    # high + molecular low level on every fragment = 2564 ORCA inputs
    assert len(calculations["high"]) + len(calculations["molecular"]) == 2564


def test_level_assignment(pilot_fragments):
    per_level = {}
    for f in pilot_fragments.selected():
        key = (f.level, f.order)
        per_level[key] = per_level.get(key, 0) + 1
    assert per_level == {
        ("periodic", 1): 64,
        ("periodic", 2): 154,
        ("molecular", 2): 237,
        ("molecular", 3): 559,
    }


def test_names_molecules_images_coordinates(pilot, pilot_fragments):
    """Identical to the prototype's frags.json: names in the same order, the
    molecules, the images, the coordinates (1e-6 Å), which pairs are outer
    pairs, which fragments go to VASP and each triple's sub-pairs."""
    assert pilot_fragments.names == [m["name"] for m in pilot.meta]
    for m in pilot.meta:
        f = pilot_fragments[m["name"]]
        assert list(f.molecules) == m["mols"]
        if f.order == 2:
            assert list(f.images[1]) == m["image"]
            assert f.in_sum == m["in_sum"]
            assert f.distances[(0, 1)] == pytest.approx(m["r_OO"], abs=1e-9)
        elif f.order == 3:
            assert [list(i) for i in f.images] == m["images"]
            pairs = [name for name, slots in f.subfragments if len(slots) == 2]
            assert pairs == m["pairs"]
        assert (f.level == "periodic") == m["vasp"]
        assert np.abs(f.coordinates - pilot.positions(m["name"])).max() < 1e-6


@pytest.fixture(scope="module")
def labels(pilot, pilot_system, pilot_fragments):
    high, periodic, molecular = pilot.results()
    correction = seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    return seamm_mbe.assemble(
        pilot_system,
        correction,
        pilot.cell_terms(),
        offsets={"water": DFE_PER_WATER},
    )


def test_pressures(pilot, labels):
    expected = pilot.expected
    debug = expected["debug"]
    breakdown = labels.breakdown
    assert breakdown["MBE"]["pressure"] == pytest.approx(
        expected["P_correction_atm"], abs=1e-6
    )
    assert breakdown["D4"]["pressure"] == pytest.approx(debug["P_d4_atm"], abs=1e-6)
    # The VASP term differs only by the prototype's truncated kB -> atm factor
    vasp = breakdown["VASP r2SCAN"]["pressure"]
    assert vasp * PROTOTYPE_KB_ATM / EXACT_KB_ATM == pytest.approx(
        debug["P_vasp_atm"], abs=1e-6
    )
    shift = vasp - debug["P_vasp_atm"]
    assert abs(shift) < 0.02
    assert labels.pressure == pytest.approx(
        expected["P_atom_conf_atm"] + shift, abs=1e-6
    )
    assert labels.molecular_pressure == pytest.approx(
        expected["P_mol_conf_atm"] + shift, abs=1e-6
    )
    # The design document's rounded targets
    assert round(labels.molecular_pressure, 1) == 1795.9
    assert round(labels.pressure, 1) == -63009.7
    assert round(breakdown["MBE"]["pressure"], 1) == 832.4
    assert round(breakdown["D4"]["pressure"], 1) == -2322.5


def test_per_body(pilot, labels):
    debug = pilot.expected["debug"]
    n = 64
    for order in (1, 2, 3):
        body = labels.per_body[order]
        assert body["pressure"] == pytest.approx(
            debug["P_per_body_atm"][order - 1], abs=1e-6
        )
        assert body["energy"] * PROTOTYPE_KJ / n == pytest.approx(
            debug["dE_per_body_kJmol"][order - 1], abs=1e-6
        )
    assert labels.per_body[2]["energy"] * EV_TO_KJ_PER_MOL / n == pytest.approx(
        5.961, abs=5e-4
    )
    assert labels.per_body[3]["energy"] * EV_TO_KJ_PER_MOL / n == pytest.approx(
        -1.194, abs=5e-4
    )
    assert [round(labels.per_body[k]["pressure"]) for k in (1, 2, 3)] == [
        -1596,
        3062,
        -634,
    ]


def test_energy_and_forces(pilot, labels):
    expected = pilot.expected
    assert labels.reference_energy == pytest.approx(expected["REF_energy_eV"], abs=1e-8)
    assert labels.energy == pytest.approx(
        expected["REF_energy_eV"] - 64 * DFE_PER_WATER, abs=1e-8
    )
    rms = float(np.sqrt((labels.forces**2).sum(1).mean()))
    assert rms == pytest.approx(expected["F_rms_eV_A"], abs=1e-9)
    assert labels.net_force * 1000 == pytest.approx(
        expected["net_force_total_meV_A"], abs=1e-6
    )


def test_increment_net_force(pilot, labels):
    assert labels.max_increment_net_force * 1000 == pytest.approx(
        pilot.expected["max_increment_net_force_meV_A"], abs=1e-9
    )
    assert labels.max_increment_net_force * 1000 <= 1.3


def test_stress_is_minus_pressure(labels):
    volume = 12.4297**3
    assert np.allclose(labels.stress, -labels.virial / volume)
    p = -np.trace(labels.stress) / 3 * seamm_mbe.units.EV_PER_A3_TO_ATM
    assert p == pytest.approx(labels.pressure, abs=1e-6)


def test_missing_fragment_is_refused(pilot, pilot_fragments):
    high, periodic, molecular = pilot.results()
    del high["t00_17_30"]
    molecular["d00_17"] = None
    del periodic["m05"]
    with pytest.raises(seamm_mbe.MissingFragmentsError) as error:
        seamm_mbe.mbe_correction(pilot_fragments, high, periodic, molecular)
    assert set(error.value.missing) == {
        ("t00_17_30", "high"),
        ("m05", "periodic"),
        ("d00_17", "molecular"),
    }


def test_from_molsystem(pilot):
    """The same enumeration through a molsystem configuration with perceived
    bonds."""
    from molsystem import SystemDB

    db = SystemDB(filename="file:seamm_mbe_test?mode=memory&cache=shared")
    try:
        system = db.create_system()
        configuration = system.create_configuration(periodicity=3)
        configuration.cell.parameters = [pilot.box] * 3 + [90.0] * 3
        configuration.atoms.append(
            x=pilot.X[:, 0],
            y=pilot.X[:, 1],
            z=pilot.X[:, 2],
            symbol=["O", "H", "H"] * 64,
        )
        configuration.coordinate_system = "Cartesian"
        configuration.perceive_bonds()
        mbe = seamm_mbe.System.from_configuration(configuration)
    finally:
        db.close()
    reference = water_system(pilot.X, pilot.box)
    assert [m.atoms.tolist() for m in mbe.molecules] == [
        m.atoms.tolist() for m in reference.molecules
    ]
    fragments = seamm_mbe.enumerate_fragments(mbe)
    assert fragments.names == [m["name"] for m in pilot.meta]
