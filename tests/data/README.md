# Test data: the pilot frame `water64_frame`

The regression oracle for `seamm_mbe` is the prototype's **pilot frame**: a
64-water OPLS-AA configuration (ChemAI job 5160, NVT, 0.997 g/cm³, cubic
L = 12.4297 Å) labelled on TinkerCliffs by the prototype pipeline in
`/projects/seamm/psaxe/periodic/tools/` (2026-09-29).

## Where it came from

- Fragment list: `frames/pilot_opls0997/frags.json` (written by
  `gen_stage2_frame.py`, 2026-09-29 10:21).
- VASP outputs: `frames/pilot_opls0997/vasp.tar.gz` (members dated
  2026-09-29 10:21 to 2026-09-30 05:32; VASP 6.6.1 gamma-only, r2SCAN,
  hard PAW O_h/H_h, 1200 eV, cell NGX 150, fragments grid-registered).
- ORCA outputs: `frames/pilot_opls0997/orca.tar.gz` (members dated
  2026-09-29 10:21 to 11:01; ORCA 6.1.1, revDSD-PBEP86-D4/2021 and
  r2SCAN-D4, def2-TZVPPD, DEFGRID3, TIGHTSCF).

All three paths are under `/projects/seamm/psaxe/periodic/` on TinkerCliffs.
They were copied read-only.

**The notebook copies are not the oracle.**
`~/Sites/mlff-training/2026-09-27_vasp-periodic-route/data/`
(`vasp_results.json.gz`, `stage2_orca.tar.gz`) holds the earlier stage-2 and
stage-2h runs of the same geometry. `assemble_frame.py` on that data gives
P_mol +1796.35 and P_MBE +831.78 atm, not the pilot's +1795.95 and +832.38.
Its energies agree to 1e-8.

## Files

- `make_pilot_fixture.py` builds the fixture. It needs the prototype's
  `vasp_io.py`, the dftd4 library and the unpacked pilot outputs; the tests
  never run it. Run 2026-10-03 with dftd4 4.2.0.
- `water64_pilot.npz` holds the per-fragment results exactly as
  `assemble_frame.py` consumed them:
  - **Units:** eV and eV/Å, converted from ORCA's Hartree and Hartree/bohr
    with the prototype's constants.
  - **Ordering:** VASP forces are reordered to molecule order. The arrays
    are concatenated over fragments in `frags.json` order, with the
    fragment metadata (`name`, `mols`, `images`, `in_sum`, `r_OO`, `vasp`,
    `pairs`) in the JSON string `meta`.
  - **Arrays:**
    - `pos`: fragment coordinates (Å).
    - `high_*`: revDSD results.
    - `molecular_*`: ORCA r2SCAN-D4 results.
    - `periodic_*`: VASP r2SCAN without D4. Fragments not run in VASP hold
      NaN.
    - `d4_*`: the dftd4 r2SCAN-D4 add-on for the VASP fragments.
    - `cell_vasp_*`: the cell's energy, forces, and stress (VASP's `in kB`
      line, XX YY ZZ XY YZ ZX, positive pushing outward).
    - `cell_d4_*`: the cell's D4, with `cell_d4_dEdstrain`, the dftd4
      "virial" dE/d(strain) in eV.
    - `X`, `box`: the frame.
- `water64_pilot_expected.json` holds the oracle's numbers: the `qc.json` of
  `assemble_frame.py` rerun on the pilot outputs (2026-10-03), plus the
  counts. It also records the route-B D4 numbers from the pilot's `qc.json`
  (`subtract_d4.py`, xnn D4). Those are a target for the Dispersion step, not
  for this library.

## One known difference

The prototype converted VASP's kB to atm with 986.923. The exact factor is
1e8 / 101325 = 986.92327, which `seamm_mbe.units` uses. The VASP cell
pressure therefore differs by a factor of 986.923 / 986.92327: 0.017 atm on
−61,519.5 atm. P_atom and P_mol move by the same 0.017 atm, and the
regression test asserts exactly that. Every other number agrees to rounding
error.
