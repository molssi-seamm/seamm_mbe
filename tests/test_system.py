"""Molecules, their types, charges and whole-molecule coordinates."""

import random

import numpy as np
import pytest

import seamm_mbe
from seamm_mbe.elements import MASS, SYMBOLS

from .helpers import water_system

# Hand-written graphs, independent of the catalog's templates:
# (symbols, bonds, index of the expected designated atom)
EC = (
    ["O", "C", "O", "O", "C", "C", "H", "H", "H", "H"],
    [(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (5, 3), (4, 6), (4, 7), (5, 8), (5, 9)],
    1,
)
FEC = (
    ["O", "C", "O", "O", "C", "C", "F", "H", "H", "H"],
    [(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (5, 3), (4, 6), (4, 7), (5, 8), (5, 9)],
    1,
)
DMC = (
    ["C", "O", "C", "O", "O", "C"] + ["H"] * 6,
    [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)]
    + [(0, 6), (0, 7), (0, 8), (5, 9), (5, 10), (5, 11)],
    2,
)
EMC = (
    ["C", "C", "O", "C", "O", "O", "C"] + ["H"] * 8,
    [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)]
    + [(0, 7), (0, 8), (0, 9), (1, 10), (1, 11), (6, 12), (6, 13), (6, 14)],
    3,
)
BF4 = (["F", "F", "B", "F", "F"], [(2, 0), (2, 1), (2, 3), (2, 4)], 2)
PF6 = (["F"] * 3 + ["P"] + ["F"] * 3, [(3, k) for k in (0, 1, 2, 4, 5, 6)], 3)
LI = (["Li"], [], 0)
WATER = (["H", "O", "H"], [(1, 0), (1, 2)], 1)

EXPECTED = {
    "EC": (EC, 0),
    "FEC": (FEC, 0),
    "DMC": (DMC, 0),
    "EMC": (EMC, 0),
    "BF4-": (BF4, -1),
    "PF6-": (PF6, -1),
    "Li+": (LI, 1),
    "water": (WATER, 0),
}


def build(parts, spacing=10.0, **kwargs):
    """A cluster from (symbols, bonds, designated) parts, placed apart."""
    symbols, bonds, coordinates = [], [], []
    rng = np.random.default_rng(1)
    for k, (s, b, _) in enumerate(parts):
        start = len(symbols)
        symbols += s
        bonds += [(start + i, start + j) for i, j in b]
        coordinates += list(rng.normal(size=(len(s), 3)) + [spacing * k, 0, 0])
    return seamm_mbe.System(symbols, coordinates, bonds=bonds, **kwargs)


def permuted(part, seed):
    """The same molecule with its atoms in a random order."""
    symbols, bonds, designated = part
    order = list(range(len(symbols)))
    random.Random(seed).shuffle(order)
    where = {old: new for new, old in enumerate(order)}
    return (
        [symbols[i] for i in order],
        [(where[i], where[j]) for i, j in bonds],
        where[designated],
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_catalog_types(name):
    """Each catalog molecule is recognized whatever its atom order, gets the
    catalog's charge, and its designated atom is the one expected."""
    part, charge = EXPECTED[name]
    for seed in range(3):
        p = permuted(part, seed)
        system = build([p])
        (molecule,) = system.molecules
        assert molecule.type == name
        assert system.types[name].charge == charge
        assert molecule.atoms[molecule.designated] == p[2]


def test_isomers_and_unknowns_are_told_apart():
    # 1,3,5-trioxane has DMC's formula, C3H6O3, but not its graph
    trioxane = (
        ["C", "O", "C", "O", "C", "O"] + ["H"] * 6,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        + [(0, 6), (0, 7), (2, 8), (2, 9), (4, 10), (4, 11)],
        0,
    )
    system = build([DMC, trioxane, trioxane])
    types = [m.type for m in system.molecules]
    assert types == ["DMC", "C3H6O3", "C3H6O3"]
    assert system.types["C3H6O3"].charge == 0


def test_mixture_charges_from_catalog():
    system = build([LI, BF4, WATER, EC])
    assert [m.type for m in system.molecules] == ["Li+", "BF4-", "water", "EC"]
    assert [system.types[m.type].charge for m in system.molecules] == [1, -1, 0, 0]
    assert system.type_counts() == {"Li+": 1, "BF4-": 1, "water": 1, "EC": 1}


def test_formal_charges_are_used_and_checked():
    # Li+ then BF4- with the charge on one F
    n = 1 + 5
    formal = [1] + [0, -1, 0, 0, 0]
    system = build([LI, BF4], formal_charges=formal)
    assert system.types["Li+"].charge == 1
    assert len(formal) == n
    # A formal-charge set that disagrees with the catalog is an error
    with pytest.raises(seamm_mbe.StructureError, match="Molecule 0 .*Li\\+"):
        build([LI, BF4], formal_charges=[0, 0, -1, 0, 0, 0])
    # All zero is "no formal charges" (bare xyz): the catalog is used
    system = build([LI, BF4], formal_charges=[0] * n)
    assert system.types["BF4-"].charge == -1


def test_parity_check_and_override():
    hydroxide = (["O", "H"], [(0, 1)], 0)
    with pytest.raises(seamm_mbe.StructureError, match="closed-shell \\(singlet\\)"):
        build([hydroxide])
    system = build([hydroxide], charges={"HO": -1})
    assert system.types["HO"].charge == -1
    system = build([hydroxide], multiplicities={"HO": 2})
    assert system.types["HO"].multiplicity == 2
    with pytest.raises(seamm_mbe.StructureError, match="no molecule of that type"):
        build([WATER], charges={"HO": -1})


def test_molecules_are_made_whole(pilot):
    """Wrap every atom into the cell: the molecules come back whole, each a
    lattice translation of the original."""
    box = pilot.box
    wrapped = pilot.X - box * np.floor(pilot.X / box)
    assert not np.allclose(wrapped, pilot.X)
    system = water_system(wrapped, box)
    for molecule in system.molecules:
        original = pilot.X[molecule.atoms]
        shift = molecule.coordinates - original
        assert np.allclose(shift, shift[0], atol=1e-10)
        assert np.allclose(shift[0] / box, np.round(shift[0] / box), atol=1e-10)


def test_cell_geometry():
    cell = [[10.0, 0, 0], [5.0, 8.0, 0], [0, 0, 12.0]]
    system = seamm_mbe.System(["Ar"], [[0, 0, 0]], cell)
    assert system.volume == pytest.approx(960.0)
    # widths: V / |b x c|, V / |c x a|, V / |a x b|
    assert np.allclose(system.widths, [960 / np.hypot(96, 60), 8.0, 12.0])
    # The minimum image is exact in a skewed cell: compare with a brute-force
    # search over a wide range of images
    vector = np.array([9.0, 7.5, 0.0])
    _, shortest = system.minimum_image(vector)
    brute = min(
        np.linalg.norm(vector + np.array(n) @ np.array(cell))
        for n in np.ndindex(5, 5, 5)
        for n in [np.array(n) - 2]
    )
    assert np.linalg.norm(shortest) == pytest.approx(brute)


def test_element_tables():
    assert len(MASS) == 56
    assert SYMBOLS[0] == "H" and SYMBOLS[-1] == "Og" and len(SYMBOLS) == 118
    assert MASS["O"] == 15.999 and MASS["H"] == 1.008


def test_two_unknown_isomers_get_distinct_names():
    ethanol = (
        ["C", "C", "O"] + ["H"] * 6,
        [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8)],
        0,
    )
    dimethyl_ether = (
        ["C", "O", "C"] + ["H"] * 6,
        [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (2, 6), (2, 7), (2, 8)],
        1,
    )
    system = build([ethanol, dimethyl_ether, ethanol])
    assert [m.type for m in system.molecules] == ["C2H6O", "C2H6O#2", "C2H6O"]
    # The default designated atom is the heavy atom nearest the centre of mass
    assert system.symbols[
        system.molecules[1].atoms[system.molecules[1].designated]
    ] in (
        "C",
        "O",
    )


def test_open_shell_parity():
    with pytest.raises(seamm_mbe.StructureError, match="multiplicity 2 needs an odd"):
        build([WATER], multiplicities={"water": 2})


def molsystem_configuration(db, symbols, xyz, formal=None):
    system = db.create_system()
    configuration = system.create_configuration(periodicity=0)
    xyz = np.asarray(xyz, dtype=float)
    configuration.atoms.append(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], symbol=symbols)
    if formal is not None:
        configuration.atoms.add_attribute("formal_charge", coltype="int", default=0)
        configuration.atoms["formal_charge"] = formal
    return configuration


def test_from_molsystem_charges_and_bonds():
    from molsystem import SystemDB

    symbols = ["Li", "O", "H", "H"]
    xyz = [[0, 0, 0], [2.0, 0, 0], [2.6, 0.8, 0], [2.6, -0.8, 0]]
    db = SystemDB(filename="file:seamm_mbe_test2?mode=memory&cache=shared")
    try:
        configuration = molsystem_configuration(db, symbols, xyz)
        with pytest.raises(seamm_mbe.StructureError, match="perceive_bonds"):
            seamm_mbe.System.from_configuration(configuration)
        configuration.perceive_bonds()
        system = seamm_mbe.System.from_configuration(configuration)
        assert [m.type for m in system.molecules] == ["Li+", "water"]
        assert system.types["Li+"].charge == 1
        assert not system.periodic

        configuration = molsystem_configuration(db, symbols, xyz, formal=[0, 0, 0, 0])
        configuration.perceive_bonds()
        configuration.atoms["formal_charge"] = [1, 0, 0, 0]
        system = seamm_mbe.System.from_configuration(configuration)
        assert system.types["Li+"].charge == 1
        configuration.atoms["formal_charge"] = [0, -1, 0, 0]
        with pytest.raises(seamm_mbe.StructureError, match="formal charges"):
            seamm_mbe.System.from_configuration(configuration)
    finally:
        db.close()


def test_heavy_elements_have_masses():
    system = seamm_mbe.System(["Pt"], [[0, 0, 0]])
    assert system.masses[0] == pytest.approx(195.08, abs=0.01)


def test_singular_cell_is_refused():
    with pytest.raises(seamm_mbe.StructureError, match="singular"):
        seamm_mbe.System(["Ar"], [[0, 0, 0]], np.zeros((3, 3)))
    with pytest.raises(seamm_mbe.StructureError, match="singular"):
        seamm_mbe.System(["Ar"], [[0, 0, 0]], [[5, 0, 0], [10, 0, 0], [0, 0, 5]])


def test_molecule_bonded_to_its_own_image_is_refused():
    xyz = [[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]]
    bonds = [(0, 1), (1, 2), (2, 3), (3, 0)]
    with pytest.raises(seamm_mbe.StructureError, match="infinite"):
        seamm_mbe.System(["C"] * 4, xyz, np.eye(3) * 12.0, bonds=bonds)
    # A real ring is fine
    ring = [[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [0, 1.5, 0]]
    seamm_mbe.System(["C"] * 4, ring, np.eye(3) * 12.0, bonds=bonds, charges=None)


def test_bondless_salt_from_molsystem():
    from molsystem import SystemDB

    db = SystemDB(filename="file:seamm_mbe_test3?mode=memory&cache=shared")
    try:
        configuration = molsystem_configuration(
            db, ["Li", "Cl", "Na", "Br"], [[0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0]]
        )
        system = seamm_mbe.System.from_configuration(configuration)
    finally:
        db.close()
    assert [m.type for m in system.molecules] == ["Li+", "Cl-", "Na+", "Br-"]
    assert [system.types[m.type].charge for m in system.molecules] == [1, -1, 1, -1]
