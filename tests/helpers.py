"""Helpers shared by the tests: the prototype's pilot frame (see
tests/data/README.md)."""

import json
from pathlib import Path

import numpy as np

import seamm_mbe

DATA = Path(__file__).parent / "data"

#: The prototype's energy offset to the cluster sets' DfE scale (eV per water)
DFE_PER_WATER = 2074.69325


def water_system(X, box, **kwargs):
    """A System of waters (O, H, H per molecule) with explicit O-H bonds."""
    n = len(X) // 3
    bonds = [(3 * i, 3 * i + k) for i in range(n) for k in (1, 2)]
    cell = None if box is None else np.eye(3) * box
    return seamm_mbe.System(["O", "H", "H"] * n, X, cell, bonds=bonds, **kwargs)


class Pilot:
    """The pilot frame's data, results and expected numbers."""

    def __init__(self):
        data = np.load(DATA / "water64_pilot.npz")
        self.data = data
        self.meta = json.loads(str(data["meta"]))
        self.X = data["X"]
        self.box = float(data["box"])
        self.expected = json.load(open(DATA / "water64_pilot_expected.json"))
        offsets = np.cumsum([0] + [3 * len(m["mols"]) for m in self.meta])
        self.slices = {
            m["name"]: slice(offsets[k], offsets[k + 1])
            for k, m in enumerate(self.meta)
        }
        self.index = {m["name"]: k for k, m in enumerate(self.meta)}

    def positions(self, name):
        return self.data["pos"][self.slices[name]]

    def results(self):
        """high, periodic (VASP + D4) and molecular results by name."""
        d = self.data
        high, periodic, molecular = {}, {}, {}
        for m in self.meta:
            name = m["name"]
            k, s = self.index[name], self.slices[name]
            high[name] = (d["high_E"][k], d["high_F"][s])
            molecular[name] = (d["molecular_E"][k], d["molecular_F"][s])
            if m.get("vasp"):
                periodic[name] = (
                    d["periodic_E"][k] + d["d4_E"][k],
                    d["periodic_F"][s] + d["d4_F"][s],
                )
        return high, periodic, molecular

    def cell_terms(self):
        d = self.data
        volume = self.box**3
        units = seamm_mbe.units
        return [
            seamm_mbe.CellTerm(
                "VASP r2SCAN",
                float(d["cell_vasp_energy"]),
                d["cell_vasp_forces"],
                units.vasp_stress_to_virial(d["cell_vasp_stress_kB"], volume),
            ),
            seamm_mbe.CellTerm(
                "D4",
                float(d["cell_d4_energy"]),
                d["cell_d4_forces"],
                units.dftd4_virial_to_virial(d["cell_d4_dEdstrain"]),
            ),
        ]
