seamm_mbe
=========
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/molssi-seamm/seamm_mbe/workflows/CI/badge.svg)](https://github.com/molssi-seamm/seamm_mbe/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/molssi-seamm/seamm_mbe/branch/main/graph/badge.svg)](https://codecov.io/gh/molssi-seamm/seamm_mbe/branch/main)

Many-body expansion (MBE) corrections for periodic cells and clusters.

`seamm_mbe` estimates high-level energies, forces and stress for a system the
high-level method can't treat directly. It takes a cheap calculation on the
whole system and adds many-body increments of [high − low] computed on small
isolated fragments: monomers, selected pairs and triples, general in the order.

The library does the bookkeeping and runs no calculations:

- molecule typing by formula and bond-graph topology, with charges, for water,
  the carbonates (EC, FEC, DMC, EMC), Li⁺, BF₄⁻, PF₆⁻, common monatomic ions,
  and any other molecule;
- fragment enumeration under periodic boundary conditions, with canonical keys
  and per-type-pair cutoffs;
- the "mixed" assignment of each increment to a periodic or a molecular low
  level;
- the increment algebra for energy, forces and the origin-independent virial;
- the assembled labels, with energy offsets, atomic and molecular pressure and
  QC.

The `mbe_step` SEAMM plug-in runs the fragment calculations through the Model
Chemistry batch contract.

It generalizes the prototype that labelled 152 periodic water cells for MLFF
training (TinkerCliffs, 2026-09-29/30), and reproduces that prototype's pilot
frame from the stored fragment results (`tests/data/`).

The design is in `MBE_correction_step_design.rst` (SEAMM workspace). The
campaign notes are in `docs/developer_guide/campaigns/2026-10-03/`.

### Copyright

Copyright (c) 2026, Paul Saxe

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
