"""Build tests/data/water64_pilot.npz from the prototype's pilot frame.

Provenance only -- the tests never run this. It needs the prototype's own
readers (``vasp_io.py`` from TinkerCliffs /projects/seamm/psaxe/periodic/tools/)
and the dftd4 Python library, and the pilot frame's outputs unpacked:

    <pilot>/frags.json            (from frames/pilot_opls0997/)
    <pilot>/vasp/r2_<name>/...    (frames/pilot_opls0997/vasp.tar.gz)
    <pilot>/orca/{rev,r2}_<name>.engrad   (frames/pilot_opls0997/orca.tar.gz)

usage: make_pilot_fixture.py <tools dir> <pilot dir> <output.npz>

The numbers are stored exactly as ``assemble_frame.py`` uses them: eV and
eV/Å, converted from ORCA's Hartree and Hartree/bohr with the prototype's
constants (HA = 27.211386245988 eV, HA/bohr = 51.42208619 eV/Å), VASP forces
reordered from POSCAR order (O first) to molecule order, the cell stress as
VASP prints it (kB, XX YY ZZ XY YZ ZX, positive = pushing outward), and the
r2SCAN-D4 add-on from the dftd4 library with the cell's virial as dftd4
returns it (dE/d(strain), converted to eV).
"""

import json
import sys

import numpy as np

tools, pilot, output = sys.argv[1:4]
sys.path.insert(0, tools)
from vasp_io import outcar, reorder, d4_r2scan  # noqa: E402

HA, EVA = 27.211386245988, 51.42208619


def engrad(path):
    lines = [x.strip() for x in open(path) if not x.startswith("#")]
    n = int(lines[0])
    energy = float(lines[1]) * HA
    gradient = np.array([float(x) for x in lines[2 : 2 + 3 * n]]).reshape(n, 3)
    return energy, -gradient * EVA


def vasp(name):
    energy, forces, stress, ok, _ = outcar(f"{pilot}/vasp/r2_{name}/OUTCAR")
    if not ok:
        raise RuntimeError(f"{name}: SCF not converged")
    order = [int(x) for x in open(f"{pilot}/vasp/r2_{name}/order.txt").read().split()]
    return energy, reorder(forces, order), stress


J = json.load(open(f"{pilot}/frags.json"))
box = J["box"]
X = np.array(J["X"])
N = len(X) // 3
Z = [8, 1, 1] * N

data = {}
energy, forces, stress = vasp("cell")
e4, f4, w4 = d4_r2scan(Z, X, np.eye(3) * box)
data["cell_vasp_energy"] = energy
data["cell_vasp_forces"] = forces
data["cell_vasp_stress_kB"] = np.asarray(stress)
data["cell_d4_energy"] = e4
data["cell_d4_forces"] = f4
data["cell_d4_dEdstrain"] = np.asarray(w4) * HA

meta = []
arrays = {k: [] for k in ("pos", "high_F", "molecular_F", "periodic_F", "d4_F")}
energies = {k: [] for k in ("high_E", "molecular_E", "periodic_E", "d4_E")}
for f in J["frags"]:
    name = f["name"]
    meta.append({k: v for k, v in f.items() if k != "pos"})
    arrays["pos"].append(np.array(f["pos"]))
    E, F = engrad(f"{pilot}/orca/rev_{name}.engrad")
    energies["high_E"].append(E)
    arrays["high_F"].append(F)
    E, F = engrad(f"{pilot}/orca/r2_{name}.engrad")
    energies["molecular_E"].append(E)
    arrays["molecular_F"].append(F)
    if f.get("vasp"):
        E, F, _ = vasp(name)
        e4, f4, _ = d4_r2scan([8, 1, 1] * len(f["mols"]), np.array(f["pos"]))
    else:
        # Not a periodic-level fragment: placeholders keep the offsets aligned
        E = e4 = np.nan
        F = np.full((3 * len(f["mols"]), 3), np.nan)
        f4 = F.copy()
    energies["periodic_E"].append(E)
    arrays["periodic_F"].append(F)
    energies["d4_E"].append(e4)
    arrays["d4_F"].append(f4)

for k, v in arrays.items():
    data[k] = np.vstack(v)
for k, v in energies.items():
    data[k] = np.array(v)
data["X"] = X
data["box"] = box
data["meta"] = np.array(json.dumps(meta))
np.savez_compressed(output, **data)
print(f"{len(meta)} fragments -> {output}")
