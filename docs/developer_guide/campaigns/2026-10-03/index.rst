==============================================================
2026-10-03: the MBE step, phase 1 (the ``seamm_mbe`` library)
==============================================================

The design is ``~/Work/SEAMM/MBE_correction_step_design.rst`` (workspace root).
Its "Decisions (2026-10-03)" section is binding, and its "Phases" section is
the plan. The science context is the ``~/Sites`` lab notebook record
``mlff-training/2026-09-27_vasp-periodic-route`` and the ``~/Sites`` memory
``vasp-periodic-prototype``. The design session ("design") reviews each phase
before its PR.

.. contents::
   :local:

Phase 1 scope
=============

Phase 1 is a pure-Python library (numpy, and molsystem for the adapter only),
extracted from the prototype's ``gen_stage2_frame.py`` and ``assemble_frame.py``
(TinkerCliffs ``/projects/seamm/psaxe/periodic/tools/``). It covers:

- molecule typing by formula and topology, with a catalog for the campaign's
  molecules, per-type charges and the designated atoms;
- fragment enumeration under PBC, with canonical keys and the prototype's
  names, general in the order;
- the mixed low-level assignment;
- the increment algebra for energy, forces and the origin-independent virial;
- the assembled labels, with energy offsets, atomic and molecular pressure,
  per-body breakdown and QC;
- the unit and sign conventions.

Phase 2 (``mbe_step``) and phase 3 (the adapters) need Paul's OK.

The oracle
==========

The regression targets in the design document are those of the **pilot frame**
(``frames/pilot_opls0997`` on TinkerCliffs). The original ``assemble_frame.py``
reproduces them exactly on the Mac from the pilot's own outputs. The oracle
Python was the ``seamm-psi4`` conda env, which has dftd4 4.2.0.

The notebook's data (``~/Sites/.../data/vasp_results.json.gz`` and
``stage2_orca.tar.gz``) are the earlier stage-2/2h runs of the same geometry.
They give P_mol +1796.35 and P_MBE +831.78 atm and are **not** the oracle. The
"science" session corrected the design document's wording accordingly.

The fixture (``tests/data/water64_pilot.npz``, 569 KB) and its provenance are
described in ``tests/data/README.md``.

Results
=======

- **Enumeration** is identical to the prototype's ``frags.json``: names in the
  same order, molecules, images, coordinates (largest difference 6e-17 Å),
  which pairs are outer pairs, which fragments go to VASP, and each triple's
  sub-pairs. The counts are 64 monomers, 391 pairs < 4.5 Å, 268 outer pairs
  and 559 connected triples.
- **Calculations:** 218 periodic fragments + the cell = 219 VASP inputs;
  high + molecular on all 1282 fragments = 2564 ORCA inputs. The molecular
  level is the closure of the molecular-assigned fragments, which is
  everything for this frame.
- **Level assignment:** periodic 64 monomers + 154 pairs; molecular 237 pairs +
  559 triples.
- **Numbers**, all to rounding error (≤ 1e-9 relative):

  - P_MBE +832.381 atm, P_D4(r2SCAN) −2322.500 atm;
  - per-body P −1595.82 / +3062.44 / −634.24 atm;
  - per-body ΔE +5.961 (2-body) and −1.194 (3-body) kJ/mol per water;
  - REF_energy −203.32968869 eV, largest increment net force 1.2186 meV/Å.

- **One deliberate difference:** the prototype converted VASP's kB to atm with
  986.923; ``seamm_mbe.units`` uses the exact 1e8/101325 = 986.92327. The
  VASP term, P_atom and P_mol therefore differ by 0.017 atm: P_mol +1795.932
  vs +1795.949, P_atom −63009.665 vs −63009.649. The test asserts that the
  ratio is exactly the two constants'.
- **Speed:** enumerating the 64-water frame takes 0.08 s.

Decisions and deviations from the plan
======================================

Agreed with "design", 2026-10-03:

#. **Validity bound.** The cell's smallest perpendicular width must exceed R + D,
   where R and D are the radius and diameter a selected fragment can have
   under its rule:

   - pairs and "compact": 2c;
   - "hub" and connected triples: 3c;
   - connected 4-body: 5c.

   Contact criteria add 2 r_max per bond. The derivation is in
   :meth:`SelectionRules.check`. It refuses rather than warns. For order ≥ 3
   the enumerator also refuses two different selected fragments of the same
   molecules. For pairs it does not need to, since pair names carry
   non-minimum images. With the pair cutoff beyond L/2 and ``check()``
   bypassed, both images would be selected. "design" proposed 3c for
   "compact"; the derivation gives 2c, since all members are within c of each
   other.
#. **Names** are the prototype's. Selected triples and 4-bodies carry no images,
   which is unique by the bound. Auxiliary fragments of order ≥ 3 carry their
   images (``_x...``) in a periodic cell and nothing in a cluster. Pair images
   beyond ±1 are spelled out (``_x2_0_-1``) instead of the prototype's sign
   letters, which would have collided.
#. **Molecular level:** the closure of the molecular-assigned fragments, with
   ``calculations(molecular_everywhere=True)`` for the consistency check.
#. **Ladders:** each increment is built from sub-fragments at its own level, the
   mixed scheme. The plan said ``assign_levels`` would enforce that a
   periodic fragment's sub-fragments are periodic. That is not needed and is
   dropped: each increment is self-consistent within its own ladder, whatever
   level its sub-fragments are assigned in the sum.
#. **Charges:** a structure's formal charges, when present and not all zero,
   override the catalog. A disagreement is an error naming the molecule. The
   parity check follows seamm_bsse's wording.
#. **Stress sign:** the batch contract says only "GPa, as the program gives it",
   and seamm_mdi documents the MDI pressure convention. So every external
   stress needs an explicit ``convention`` ("pressure" or "stress"), and
   ``to_seamm`` writes σ = −P. **For phase 3:** ``vasp_step.analyze_task``
   must state its stress convention.
#. **No dftd4:** the D4 add-on is a ``CellTerm`` (or added to the periodic
   fragment results) supplied by the caller. The route-B twin belongs to the
   Dispersion step; its pilot numbers are kept in the expected-values file
   for that step.

Bugs found by the unit tests
============================

- A bond given as (i, j) with i > j was dropped from the molecule's graph,
  which mistyped it. The molsystem adapter always gives i < j, so the pilot
  regression could not see it.
- ``mbe_correction`` on fragments without levels reported nothing missing.
  It now asks for ``assign_levels`` first.

Review (2026-10-03)
===================

"design" and a review subagent read every module and verified each finding
with throwaway scripts. The verdict was to approve for the PR after these
fixes, all made, with tests:

#. **Collision guard.** The same-molecules guard ran on every *connected*
   placement before the rule filter, so "hub" and "compact" were held to the
   connected bound. Four Ar 2.9 Å apart in a 9.1 Å cell, hub 4-body, passed
   ``check()`` but the enumerator refused it. The guard now applies to the
   placements that pass the rule. The test fails on the old code.
#. **Packaging.** The wheel installed a top-level ``tests``: added
   ``include = ["seamm_mbe*"]``. The sdist lacked ``tests/data`` and
   ``docs``: fixed in ``MANIFEST.in``.
#. **Masses.** The table stops at Ba and was looked up eagerly. Masses are
   now lazy, falling back to molsystem's table for heavier elements.
   molsystem's own midpoint values (O 15.9995) would move P_mol, so the
   conventional table stays first.
#. **Cells and inputs.**

   - A singular cell is refused.
   - An empty periodic system passes ``check()``.
   - ``assign_levels`` needs real booleans: 1 means 1 Å.
#. **Cutoff tables.** Contradictory entries are refused: the same pair
   twice in ``__post_init__``, and equally specific matches at lookup.
#. **Bond-less salts.** A bond-less salt from molsystem is accepted. The
   halide ions and noble gases count as monatomic, and the catalog gained
   Na⁺, K⁺, F⁻, Cl⁻, Br⁻ and I⁻.
#. **Increments.** ``increments()`` on names not closed under sub-fragments
   gives a clear error.
#. **Images.** A molecule bonded to its own periodic image is refused. The
   ``minimum_image`` docstring no longer claims exactness for long vectors in
   non-reduced cells.

The review also verified, so there is no need to re-audit:

- triclinic cells against brute force;
- the R + D bound, including compact = 2c;
- the ions, charges and parity checks;
- the order-4 rules and the Möbius inversion, against an independent
  recursive evaluation;
- the force scatter, the virial and the ladders;
- the pressures and the units.

**Phase-2 notes:**

- ``to_seamm`` returns a (3, 3) stress, but SEAMM's stress property is
  Voigt [6], so the step converts it.
- The design document's targets now carry the exact-constant values.

Next
====

- The "design" review, then the repo ``molssi-seamm/seamm_mbe`` (Paul), the
  PR via the release-seamm-plugin skill, the release, and GitHub Pages.
- Phase 2 (``mbe_step``) with Paul's OK.
