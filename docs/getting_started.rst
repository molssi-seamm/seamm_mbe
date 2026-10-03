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
:meth:`SelectionRules.check` refuses a cell too small for the cutoffs, since
the same molecules could otherwise form two different fragments.

Corrections to selected increments
----------------------------------

``mbe_correction(..., corrections={name: (dE, dF)})`` adds a correction to a selected
fragment's increment in the sum only, never to the sub-fragment increments that
higher fragments subtract. Pairwise counterpoise works this way: with
dE_ij^CP − dE_ij for each pair, the result is E(1) + Σ dE_ij^CP + Σ dE_ijk. The
triples still subtract the uncorrected pairs, so a pair's basis-set superposition
error does not move into the 3-body terms.
