Getting Started
===============

``seamm_mbe`` is the bookkeeping behind SEAMM's many-body correction (MBE). It
estimates high-level energies, forces and stress for a periodic cell or a
large cluster as a cheap calculation of the whole system plus many-body
increments of [high − low], each computed on a small isolated fragment::

    E(cell) ≈ E_low(cell) + Σ_F dE_F

The increments come from monomers, selected pairs and selected triples
(designed for general order). The library does no calculations itself. The
``mbe_step`` plug-in runs them through the Model Chemistry batch contract and
passes the results here.

Installing
----------

::

    pip install seamm_mbe

It needs numpy, and molsystem for :meth:`System.from_configuration`.

Units
-----

Everything is in eV, Å, eV/Å (forces, not gradients) and eV for virials
(W = Σ r ⊗ f, so P = W/V). Stress σ is in eV/Å³ with σ = −P, the ASE/xnn
convention. :mod:`seamm_mbe.units` converts from SEAMM's
``analyze_task`` results (kJ/mol, kJ/mol/Å gradients, GPa) and back. Every
external stress needs an explicit ``convention``: ``"pressure"`` (VASP, MDI)
or ``"stress"`` (ASE, xnn).

Basic usage
-----------

.. code-block:: python

    import seamm_mbe

    # The system: from arrays (bonds define the molecules) ...
    system = seamm_mbe.System(symbols, xyz, cell, bonds=bonds)
    # ... or from a molsystem configuration with bonds
    system = seamm_mbe.System.from_configuration(configuration)

    # The fragments: the prototype's water rules are the default
    rules = seamm_mbe.SelectionRules(
        max_order=3,
        criterion="designated",            # water O-O
        cutoffs={2: 4.5, 3: 3.5},          # or tables by type pair
        rules={3: "connected"},
    )
    fragments = seamm_mbe.enumerate_fragments(system, rules)

    # The mixed low level: monomers and pairs < 3.5 Å periodic, the rest molecular
    seamm_mbe.assign_levels(fragments, {1: True, 2: 3.5})
    todo = fragments.calculations()   # {"high": [...], "periodic": [...], "molecular": [...]}

    # ... run them (mbe_step's job), giving {name: (energy_eV, forces_eV_per_A)} ...

    correction = seamm_mbe.mbe_correction(fragments, high, periodic, molecular)
    labels = seamm_mbe.assemble(
        system,
        correction,
        [
            seamm_mbe.CellTerm("VASP r2SCAN", e, f, seamm_mbe.units.vasp_stress_to_virial(kB, V)),
            seamm_mbe.CellTerm("D4", e4, f4, seamm_mbe.units.dftd4_virial_to_virial(dEde)),
        ],
        offsets={"water": 2074.69325},     # eV per molecule, to the training sets' scale
    )
    labels.reference_energy, labels.forces, labels.stress, labels.molecular_pressure

Molecules and types
-------------------

The molecules are the connected parts of the bond graph, made whole across
the cell boundary. Each is typed by its formula and a Weisfeiler–Lehman hash
of its bond graph. The built-in catalog has water, EC, FEC, DMC, EMC, Li⁺,
BF₄⁻, PF₆⁻ and the monatomic ions Na⁺, K⁺, F⁻, Cl⁻, Br⁻ and I⁻, each with its
charge and designated atom (water's O, the
carbonyl C, the ion's central atom). Other molecules are named by their
formula.

Charges come from the structure's per-atom formal charges when it has any
(e.g. SDF ``M  CHG``). They are checked against the catalog, and a mismatch
names the molecule. Otherwise the catalog's charges apply. An electron-count
parity check catches a charge put on the wrong molecule. Explicit charges by
type override both.

Fragment names
--------------

Names are the prototype's (``gen_stage2_frame.py``), so stored results match:

- ``m00``: a monomer.
- ``d03_41``: a pair. When the second molecule is not at its minimum image,
  the image follows, e.g. ``d03_41_x0p0``.
- ``t00_17_30`` and ``q..``: selected triples and 4-bodies.

A selected triple's or 4-body's sub-fragments that are not selected
themselves (e.g. the "outer pair" of a connected triple) are computed too,
with ``in_sum`` False. Higher-order ones carry their images in their names.

In a cell narrower than the cutoffs need for uniqueness, the same molecules can
form several different fragments through different images: in a 15.2 Å cell of
32 ethylene carbonates, molecules 0, 1 and 25 form three different connected
triples. :meth:`SelectionRules.check` then warns, and each of these fragments is
enumerated and computed on its own. The one with every molecule at its minimum
image keeps the plain name (``t00_01_25``); the others carry their images
(``t00_01_25_xm00m00``), as do their sub-pairs. A fragment never holds a
molecule twice: when molecule a is bonded to two images of b, the triple
(b, a, b') is skipped, while (a, b, k) and (a, b', k) are both enumerated.
Sub-fragments that equal another fragment up to whole cells share its key and
are computed once.

Two cases are refused, since no fragment list would be right:

- the cell is narrower than the cutoff plus the molecules' size, so the
  neighbour search could miss partners;
- a molecule is within the pair cutoff of its own image: a real interaction
  that no fragment can hold, which skipping would silently drop.

Corrections to selected increments
----------------------------------

``mbe_correction(..., corrections={name: (dE, dF)})`` adds a correction to a selected
fragment's increment in the sum only, never to the sub-fragment increments that
higher fragments subtract. Pairwise counterpoise works this way: with
dE_ij^CP − dE_ij for each pair, the result is E(1) + Σ dE_ij^CP + Σ dE_ijk. The
triples still subtract the uncorrected pairs, so a pair's basis-set superposition
error does not move into the 3-body terms.

A high level per order
----------------------

The high level can differ by order, e.g. revDSD at def2-QZVPPD for the monomers and
pairs and at def2-TZVPPD for the triples. Each order's increments are then built
entirely at that order's level: a triple's increment uses the triple, its pairs and
its monomers all at the triples' level, so those pairs and monomers are computed at
both levels::

    todo = fragments.calculations(high_levels={3: "high:3"})
    # todo["high"]: the monomers and pairs (and their sub-fragments)
    # todo["high:3"]: the triples with all their sub-pairs and monomers
    correction = seamm_mbe.mbe_correction(
        fragments, high, periodic, molecular, high_by_order={3: high_tz}
    )

Each increment records the high level it used (``Increment.high_level``). With
the same results for both levels the correction is exactly the single-level one.
