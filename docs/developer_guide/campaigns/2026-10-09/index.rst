2026-10-09: scoping an ion-shell fragmentation rule
===================================================

Scoped 2026-10-09 (Paul, via science). **Implemented 2026-10-10** in seamm_mbe
(``seamm_mbe.shells``: ``IonShellRules``, ``ion_shells``, ``UnitSystem``;
``SelectionRules.shell_max_order``) and mbe_step ('Ion shells' and its settings),
as described below. Validation checks 1-3 are still to run.

Why
---

C.3 (a 12-molecule Li⁺/BF₄⁻/6 H₂O/4 EC cluster, both geometries, against a
canonical revDSD-PBEP86-D4/def2-TZVPPD calculation) failed the Li⁺ gate with the
per-molecule expansion:

- The Li⁺ 3-body [revDSD − r2SCAN] increments sum to +19 to +23 kJ/mol, and the
  4-body ones to −5 to −18 kJ/mol.
- Moving D4 outside the expansion does not help, because the error is electronic:
  the polarisation around Li⁺ differs between the two methods.

Treating Li⁺ and its first shell as one fragment fixes it. At three units:

- cip: +1.7 kJ/mol, forces 3.6 meV/Å RMS (13 max);
- ssip: +1.1 kJ/mol, forces 3.7 meV/Å RMS (20 max);
- the per-molecule expansion gave −11.4 / −5.4 kJ/mol and 16 meV/Å RMS (84–95
  max);
- the remaining shell 3-body terms are at most 0.5 kJ/mol, and the 4-body terms
  at most 0.3;
- at two units: +0.2 / −2.6 kJ/mol, 5.7 / 6.0 meV/Å RMS.

The scripts and outputs are in TinkerCliffs ``/projects/seamm/psaxe/mbe_c2/c3``:
``c3_d4_rescore.py``, ``c3_shell_driver.py`` and ``c3_shell_score.py``.

1. The rule
-----------

**Units.** An ion together with every molecule that has an atom within the shell
cutoff of it forms one **unit**. All other molecules are units by themselves. The
units are rebuilt for each frame, and the expansion runs over units exactly as it
now runs over molecules:

- the contact-distance selection;
- pairs and connected triples;
- canonical keys and periodic images;
- the inclusion–exclusion algebra.

The shell members are placed at the image nearest the ion, as atoms are made whole
within a molecule.

**Cutoffs.**

- They are per ion type and per partner element, from the first minimum of g(r).
- C.3's shells sit at Li–O 1.83–2.05 Å and Li–F 1.85 Å. The next molecule is at
  3.0 Å or more.
- Proposed: Li⁺–O 2.6 Å (carbonyl and ether O of EC, FEC and DMC, and water O) and
  Li⁺–F 2.6 Å (BF₄⁻, PF₆⁻). Check these against the g(r) of the target liquids
  when frames exist. Literature Li–O first minima in EC/DMC are 2.6–2.8 Å.

**Which ions.**

- Li⁺ only.
- The anions don't need it. In C.3 the BF₄⁻-only increments were small: 3-body −3
  kJ/mol summed, at most 0.5 per term.
- Na⁺/Cl⁻ should keep the NaCl scheme, ion tetramers with D4 by the hybrid rule. It
  is validated, all 132 cells are labelled, and training has started. Changing it
  would only make it inconsistent.

**Truncation.**

- Units up to triples, as now (n = 3 beats n = 4 in C.3).
- A cheaper option is to truncate fragments that hold a shell at pairs (C.3 at n =
  2: ~6 meV/Å RMS, about the carbonate scheme's truncation floor). See `4. Cost`_.

2. Shared molecules
-------------------

A molecule can lie in two shells, as a solvent molecule bridging two Li⁺, or as an
anion bridging them in an aggregate.

- **Nearest ion (recommended).** Each molecule joins the shell of the nearest ion
  within the cutoff, so units never overlap, and their size stays about one ion
  plus four. A bridging molecule's interaction with the second ion is then a unit
  pair, which is computed and is the per-molecule treatment of that contact.
- **Merge.** The two shells become one unit of about 90 atoms. Its pairs and
  triples, at about 100–115 atoms, can't be afforded at QZ or TZ revDSD gradients.
  Reject it.
- **Cap.** At most k members per shell (k = 5), the nearest first, with the rest as
  separate units. This is only a guard against rare crowded shells, combined with
  the nearest-ion rule.

A contact ion pair is already handled: the anion is in the Li⁺ shell, as in C.3's
cip, which passed. Anions get no shells, so an anion is never in two units.

**The size distribution on real cells is not measured yet.** No 1 m LiPF₆ in EC:DMC
frames exist. Science's plan carves them from classical runs that haven't been made.
At 1 mol/kg, about 11 solvent molecules per Li⁺, the literature describes mostly
separated ion pairs, contact pairs in a minority, coordination numbers 4–5, and Li–Li
distances mostly over 7 Å. So shared shells should be rare in a cell of about 400
atoms with three Li⁺. This needs checking on frames: shell sizes, the fraction of
molecules claimed by two ions, and the effect of the cap.

3. Interface
------------

**seamm_mbe.**

- In ``System``, after the molecules are found: a grouping step builds the units
  from ``SelectionRules.ion_shells``, a table {ion type: {partner element:
  cutoff}}, with ``shared = "nearest"`` and ``max_shell = 5``.
- The rest of seamm_mbe works on units in place of molecules, so ``fragments.py``
  and ``algebra.py`` don't change.
- A unit's type is its composition, e.g. ``Li+(EC)3(DMC)``. Its charge, its
  multiplicity and its energy offset (DfE0) are the sums over its members.
- The cutoff tables need a wildcard or an "ion-shell" class for these types.
- Names keep the ``m/d/t`` scheme over unit indices, so units must be numbered
  deterministically, e.g. shells first in ion order.

**mbe_step parameters.**

- "ion shells": none (the default) or a table, with Li⁺–O/F 2.6 Å as the preset;
- "shared shell molecules": nearest or cap;
- "shell truncation": triples or pairs.

**Low levels: the open problem.**

- In option A the 1-body low level is VASP, a monomer in a cell-sized box. A Li⁺
  shell is a charged unit (+1, or 0 for a contact pair), and its VASP energy in a
  box carries a Madelung term that depends on the box: about 1.4 eV for +1 in a
  15 Å cube.
- The NaCl pipeline avoids this with a constant infinite-box energy per bare ion.
  That can't work for a shell, whose geometry changes with every frame.
- Options:

  - VASP with a monopole correction (LMONO / Makov–Payne, residual of order L⁻³);
  - ORCA r2SCAN as the low level for every fragment that holds a shell. The shell
    unit's 1-body term would then also need an ORCA − VASP r2SCAN difference, to tie
    it to the VASP cell.

- This needs a periodic validation, as the NaCl pilot did, before production.

**Labels and caching.**

- Each frame is its own job with its own task directories, and nothing is shared
  between frames. So units that change between frames don't interact with
  caching.
- A frame's labels, energy, forces and stress, come from one consistent
  decomposition.
- Across frames, a molecule moving through the cutoff changes the decomposition, so
  the truncation error jumps by up to the difference between the two schemes,
  about 1–2 kJ/mol here. To the fit that is per-frame noise, smaller than the
  scheme's error. Forces are unaffected within a frame.

4. Cost
-------

**Per-task cost.** Measured from C.3's 1,048 bundles on TinkerCliffs (sacct CPU
time per task, 4–8 ranks):

- revDSD/TZ gradient ≈ 0.4, 1.9, 8.3, 19 and 39 core-h at 20, 30, 45, 57 and 69
  atoms (∝ N^3.6);
- r2SCAN/TZ ∝ N^2.3, at most a few core-h;
- QZ costs about 3–5× TZ: the EC pilot's QZ pairs averaged 2.3 core-h per task,
  and its TZ triples 2.4.

**The reference point.** The 32-EC pilot frame measured **5,163 core-h** on TC (the
~1,700 figure was an early estimate):

- high-level TZ triples, with their pairs and monomers: 3,750;
- high-level QZ pairs and monomers: 633;
- low-level triples: 389;
- cell: 142;
- periodic monomers: 133;
- low-level pairs and monomers: 116.

**Assumed cell** (until frames exist): 1 m LiPF₆ in EC:DMC with about 400 atoms,
three Li⁺ and about 34 solvent molecules. Each shell has about 45 atoms (Li⁺ + 4
carbonates), about 14 neighbouring units within the 4.5 Å contact cutoff and about
60 connected triples. That gives about 42 shell pairs (~57 atoms) and about 180
shell triples (~69 atoms) per cell.

=====================================================  ==================
scheme                                                 core-h per cell
=====================================================  ==================
shell pairs QZ + shell triples TZ (as EC/FEC)          ~15,000 (≈ 3× EC)
shell pairs TZ + shell triples TZ                      ~12,500
shell pairs QZ, no shell triples                       ~7,000
shell pairs TZ, no shell triples                       ~4,500 (≈ EC)
=====================================================  ==================

In each scheme, the solvent-only fragments (about 3,600) and the low levels (+5–10%)
are included. The shell triples dominate: 180 × 39 core-h ≈ 7,000. Truncating the
shell at pairs is the main saving, and C.3 suggests it costs little accuracy (about
6 meV/Å RMS). That should be validated on the target composition first.

5. Validation beyond C.3
------------------------

1. **A target-composition cluster:** Li⁺ with a PF₆⁻ contact pair and EC/DMC
   shells, no water. C.3 used BF₄⁻ and water. Score it at n = 2 and 3 units against
   canonical revDSD, as C.3 was.
2. **A two-Li⁺ cluster with a shared (bridging) carbonate,** to test the
   nearest-ion rule against merging, and both against canonical.
3. **The charged shell's 1-body term in a periodic cell:** VASP in a box with a
   monopole correction against ORCA at an infinite box, on a few shells of
   different charge (+1 separated, 0 contact pair).
4. **One periodic pilot cell** of the target electrolyte, end to end, with the cost
   per level measured, as for the EC and FEC pilots.
5. **The g(r) cutoffs and shell statistics** on the frames once they exist: the
   size distribution, the shared fraction, and the effect of the cap.

Steps 1, 2 and 4 need canonical revDSD references for 50- to 64-atom clusters.
C.3's canonical runs used 32 ranks on an Owl Genoa node, about 12 hours each.
