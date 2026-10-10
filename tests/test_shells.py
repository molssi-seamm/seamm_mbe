"""Ion shells: an ion and its first-shell molecules as one unit."""

import itertools

import numpy as np
import pytest

import seamm_mbe

#: A water's H atoms relative to its O (Å), pointing away from +x
H_OFFSETS = np.array([[0.59, 0.76, 0.0], [0.59, -0.76, 0.0]])


def rotation_to(direction):
    """A rotation taking +x to ``direction`` (so a water's H point away from it)."""
    x = np.array([1.0, 0.0, 0.0])
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    v = np.cross(x, d)
    s, c = np.linalg.norm(v), float(x @ d)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * k + (1 - c) * k @ k


def build(ions, waters, cell=None):
    """A System of Li⁺ ions and waters. ``waters`` are (O position, direction
    the H atoms point away from)."""
    symbols, xyz, bonds = [], [], []
    for p in ions:
        symbols.append("Li")
        xyz.append(p)
    for o, away in waters:
        n = len(symbols)
        rot = rotation_to(np.asarray(o, dtype=float) - np.asarray(away, dtype=float))
        symbols += ["O", "H", "H"]
        xyz += [o, *(np.asarray(o) + H_OFFSETS @ rot.T)]
        bonds += [(n, n + 1), (n, n + 2)]
    return seamm_mbe.System(symbols, np.array(xyz, dtype=float), cell, bonds=bonds)


def around(center, distances, start=0.0):
    """Waters whose O sit at the given distances from ``center``, spread out."""
    center = np.asarray(center, dtype=float)
    out = []
    for k, d in enumerate(distances):
        theta = start + k * 2 * np.pi / len(distances)
        tilt = 0.6 * (-1) ** k
        u = np.array([np.cos(theta), np.sin(theta), tilt])
        u /= np.linalg.norm(u)
        out.append((center + d * u, center))
    return out


def test_a_shell_is_one_unit():
    far = [((7.0, 0.0, 0.0), (0, 0, 0)), ((0.0, 8.0, 0.0), (0, 0, 0))]
    system = build([(0.0, 0.0, 0.0)], around((0, 0, 0), [1.9, 1.95, 2.0, 2.05]) + far)
    units = seamm_mbe.ion_shells(system)
    assert len(units.molecules) == 3
    assert units.shells == {0}
    assert units.members == [(0, 1, 2, 3, 4), (5,), (6,)]
    shell = units.molecules[0]
    assert shell.type == "Li+[water4]"
    assert units.types[shell.type].charge == 1
    assert units.types[shell.type].formula == "H8LiO4"
    assert list(shell.atoms) == list(range(13))
    assert np.allclose(shell.coordinates, system.coordinates[:13])
    assert shell.atoms[shell.designated] == 0  # the ion
    # The other units are the molecules themselves, renumbered
    assert units.molecules[1].type == "water"
    assert list(units.molecules[2].atoms) == list(system.molecules[6].atoms)
    assert units.type_counts() == {"Li+[water4]": 1, "water": 2}
    assert units.shell_sizes() == {0: 4}


def test_a_shared_molecule_joins_the_nearer_ion():
    ions = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    shared = ((1.8, 0.0, 0.0), (0.0, 0.0, 0.0))  # 1.8 Å from the first, 2.2 Å
    waters = [shared, ((-1.9, 0.0, 0.0), (0, 0, 0)), ((5.9, 0.0, 0.0), (4, 0, 0))]
    units = seamm_mbe.ion_shells(build(ions, waters))
    assert units.members == [(0, 2, 3), (1, 4)]
    assert units.types[units.molecules[0].type].charge == 1
    # Ions never join a shell, so the two Li⁺ are in different units
    assert all(sum(m in (0, 1) for m in members) == 1 for members in units.members)


def test_the_cap_keeps_the_nearest_members():
    distances = [1.9, 2.0, 2.1, 2.2, 2.3, 2.45]
    system = build([(0.0, 0.0, 0.0)], around((0, 0, 0), distances))
    units = seamm_mbe.ion_shells(system)
    assert units.shell_sizes() == {0: 5}
    assert units.members[0] == (0, 1, 2, 3, 4, 5)
    assert units.members[1] == (6,)  # the furthest, at 2.45 Å
    unlimited = seamm_mbe.ion_shells(system, seamm_mbe.IonShellRules(max_members=None))
    assert unlimited.shell_sizes() == {0: 6}


def test_the_cutoffs_are_per_element():
    system = build([(0.0, 0.0, 0.0)], around((0, 0, 0), [1.9, 2.4]))
    units = seamm_mbe.ion_shells(system, seamm_mbe.IonShellRules({"Li": {"O": 2.2}}))
    assert units.members == [(0, 1), (2,)]
    # No ion of a listed element: the units are the molecules
    none = seamm_mbe.ion_shells(system, seamm_mbe.IonShellRules({"Na": {"O": 3.0}}))
    assert none.members == [(0,), (1,), (2,)]
    assert none.shells == frozenset()


def test_bad_rules_are_refused():
    with pytest.raises(seamm_mbe.ShellError, match="nearest"):
        seamm_mbe.IonShellRules(shared="merge")
    with pytest.raises(seamm_mbe.ShellError, match="positive"):
        seamm_mbe.IonShellRules({"Li": {"O": 0.0}})


def test_a_shell_across_the_cell_boundary_is_whole():
    box = 12.0
    cell = np.eye(3) * box
    li = (0.4, 6.0, 6.0)
    # O at x = 10.5 is 1.9 Å from the ion through the boundary
    waters = [((10.5, 6.0, 6.0), (12.4, 6.0, 6.0)), ((6.0, 6.0, 6.0), (6.0, 6.0, 7.0))]
    system = build([li], waters, cell)
    units = seamm_mbe.ion_shells(system)
    assert units.members == [(0, 1), (2,)]
    shell = units.molecules[0]
    o = list(shell.atoms).index(1)
    assert np.linalg.norm(
        shell.coordinates[o] - shell.coordinates[shell.designated]
    ) == (pytest.approx(1.9))
    assert np.allclose(shell.coordinates[o], (10.5 - box, 6.0, 6.0))


def test_without_ions_the_fragments_are_the_same(pilot_system):
    rules = seamm_mbe.SelectionRules(criterion="designated")
    plain = seamm_mbe.enumerate_fragments(pilot_system, rules)
    units = seamm_mbe.ion_shells(pilot_system)
    assert units.shells == frozenset()
    via_units = seamm_mbe.enumerate_fragments(units, rules)
    assert via_units.names == plain.names
    for a, b in zip(plain, via_units):
        assert np.array_equal(a.atoms, b.atoms)
        assert np.allclose(a.coordinates, b.coordinates)


def shell_cluster():
    """Li⁺ with four waters in its shell and three other waters."""
    others = [
        ((6.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        ((0.0, 6.5, 0.0), (0.0, 5.0, 0.0)),
        ((0.0, 0.0, 7.0), (0.0, 0.0, 5.0)),
    ]
    return build([(0.0, 0.0, 0.0)], around((0, 0, 0), [1.9, 1.95, 2.0, 2.05]) + others)


def test_shell_truncation_drops_only_the_shells_triples():
    units = seamm_mbe.ion_shells(shell_cluster())
    rules = seamm_mbe.SelectionRules(
        max_order=3, criterion="contact", cutoffs={2: 100.0, 3: 100.0}
    )
    full = seamm_mbe.enumerate_fragments(units, rules)
    assert full.counts() == {
        1: {"selected": 4, "auxiliary": 0},
        2: {"selected": 6, "auxiliary": 0},
        3: {"selected": 4, "auxiliary": 0},
    }
    rules.shell_max_order = 2
    truncated = seamm_mbe.enumerate_fragments(units, rules)
    assert truncated.counts()[2] == {"selected": 6, "auxiliary": 0}
    assert [f.molecules for f in truncated.by_order(3)] == [(1, 2, 3)]
    rules.shell_max_order = 1
    monomer_only = seamm_mbe.enumerate_fragments(units, rules)
    assert sorted(f.molecules for f in monomer_only.by_order(2)) == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def pair_energy(xyz, charges):
    energy, forces = 0.0, np.zeros_like(xyz)
    for i, j in itertools.combinations(range(len(xyz)), 2):
        d = xyz[i] - xyz[j]
        r = np.linalg.norm(d)
        e = charges[i] * charges[j] / r
        energy += e
        forces[i] += e * d / r**2
        forces[j] -= e * d / r**2
    return energy, forces


def test_a_pairwise_model_is_exact_at_two_units():
    """With an atom-pair energy the expansion over units is exact at pairs:
    the shell's own pairs are inside its 1-body term."""
    system = shell_cluster()
    units = seamm_mbe.ion_shells(system)
    rules = seamm_mbe.SelectionRules(
        max_order=2, criterion="contact", cutoffs={2: 100.0}
    )
    fragments = seamm_mbe.enumerate_fragments(units, rules)
    seamm_mbe.assign_levels(fragments)
    q = np.random.default_rng(5).normal(size=len(system.symbols))
    high = {f.name: pair_energy(f.coordinates, q[f.atoms]) for f in fragments}
    low = {f.name: (0.0, np.zeros((f.n_atoms, 3))) for f in fragments}
    correction = seamm_mbe.mbe_correction(fragments, high, molecular=low)
    energy, forces = pair_energy(system.coordinates, q)
    assert correction.energy == pytest.approx(energy)
    assert np.allclose(correction.forces, forces)


def test_shells_need_a_contact_criterion():
    units = seamm_mbe.ion_shells(shell_cluster())
    for criterion in ("designated", "com", "cog"):
        rules = seamm_mbe.SelectionRules(criterion=criterion)
        with pytest.raises(seamm_mbe.SelectionError, match="contact criterion"):
            seamm_mbe.enumerate_fragments(units, rules)
    rules = seamm_mbe.SelectionRules(criterion="heavy contact")
    seamm_mbe.enumerate_fragments(units, rules)
