"""Which fragments are selected: the distance criterion, the cutoffs and the
rules for each order.
"""

from dataclasses import dataclass, field

import numpy as np

#: How the distance between two molecules is measured
CRITERIA = ("designated", "com", "cog", "contact", "heavy contact")

#: The rules for fragments of order 3 and higher
RULES = ("none", "connected", "hub", "compact")


class SelectionError(ValueError):
    """The selection is not well defined for this system."""


def _default_cutoffs():
    return {2: 4.5, 3: 3.5}


def _default_rules():
    return {3: "connected"}


@dataclass
class SelectionRules:
    """The rules selecting the fragments.

    Attributes
    ----------
    max_order : int
        The highest order of fragment (1 = monomers only, 2 = pairs, ...).
    criterion : str
        The intermolecular distance (Å): "designated" (between the types'
        designated atoms, e.g. water O--O), "com" or "cog" (centres of mass or
        geometry), "contact" (closest atom--atom contact) or "heavy contact"
        (closest contact between non-hydrogen atoms).
    cutoffs : {int: float or {(str, str): float}}
        Per order, the cutoff (Å): one value, or a table by molecule-type pair.
        A table's keys are (type, type) tuples in either order; "*" matches any
        type, and the most specific entry wins. For pairs the cutoff selects
        the pairs; for order n >= 3 it defines the bonds of the connectivity
        graph that the order's rule tests.
    rules : {int: str}
        Per order n >= 3: "none" (no fragments of that order), "connected"
        (the n molecules form a connected graph), "hub" (one molecule is
        bonded to all the others) or "compact" (all of them bonded to each
        other). For n = 3, "connected" and "hub" are the same.
    """

    max_order: int = 3
    criterion: str = "designated"
    cutoffs: dict = field(default_factory=_default_cutoffs)
    rules: dict = field(default_factory=_default_rules)

    def __post_init__(self):
        if self.criterion not in CRITERIA:
            raise SelectionError(
                f"Unknown distance criterion {self.criterion!r}: one of {CRITERIA}"
            )
        if self.max_order < 1:
            raise SelectionError("The maximum order must be at least 1")
        for n in range(2, self.max_order + 1):
            if n not in self.cutoffs:
                raise SelectionError(f"No cutoff given for order {n}")
            if n >= 3:
                rule = self.rules.get(n)
                if rule not in RULES:
                    raise SelectionError(
                        f"The rule for order {n} is {rule!r}; it must be one of "
                        f"{RULES}"
                    )
            values = self.cutoffs[n]
            if isinstance(values, dict):
                seen = {}
                for (a, b), value in values.items():
                    pair = tuple(sorted((a, b)))
                    if seen.setdefault(pair, float(value)) != float(value):
                        raise SelectionError(
                            f"The order-{n} cutoff table gives the pair {pair} "
                            f"both {seen[pair]} and {float(value)} Å."
                        )
            values = values.values() if isinstance(values, dict) else [values]
            if any(float(v) <= 0 for v in values):
                raise SelectionError(f"The cutoffs for order {n} must be positive")

    def rule(self, order):
        """The rule for an order: "pair" for 2, else the configured rule."""
        if order == 2:
            return "pair"
        return self.rules[order]

    def cutoff(self, order, type_a, type_b):
        """The cutoff (Å) of an order for a pair of molecule types."""
        table = self.cutoffs[order]
        if not isinstance(table, dict):
            return float(table)
        matches = {}
        for (a, b), value in table.items():
            for x, y in ((a, b), (b, a)):
                if x in (type_a, "*") and y in (type_b, "*"):
                    score = (x != "*") + (y != "*")
                    matches.setdefault(score, set()).add(float(value))
        if not matches:
            raise SelectionError(
                f"The order-{order} cutoff table has no entry for the pair "
                f"({type_a}, {type_b}); add one, or a '*' default."
            )
        values = matches[max(matches)]
        if len(values) > 1:
            raise SelectionError(
                f"The order-{order} cutoff table is ambiguous for the pair "
                f"({type_a}, {type_b}): equally specific entries give "
                f"{sorted(values)} Å."
            )
        return values.pop()

    def reach(self, order):
        """The bound on the radius R and diameter D of a selected fragment of
        an order, in units of its cutoff (see :meth:`check`)."""
        rule = self.rule(order)
        if rule in ("pair", "compact"):
            return 1, 1
        if rule == "hub":
            return 1, 2
        return order // 2, order - 1  # connected

    def check(self, system):
        """Raise :class:`SelectionError` unless the selection is well defined.

        A fragment is named by its molecules (and, for a pair, the image), so
        two different physical fragments made of the same molecules would
        collide and one would be silently lost. This never warns: it refuses.

        The bound. Let a selected fragment of order n have, under its rule,
        radius at most R (some member A is within R of every member) and
        diameter at most D (no two members further apart than D), measured
        with the criterion's distance. Suppose two different selected
        fragments F1 and F2 contain the same molecules. Put both with
        molecule A, F1's centre, at the same position. Some member k then sits
        at x in F1 and at x + T in F2, T a non-zero lattice vector, so::

            length(T) <= d1(A, k) + d2(A, k) <= R + D.

        Every non-zero lattice vector is at least as long as the smallest
        perpendicular width of the cell, L_min. So L_min > R + D makes the
        collision impossible. With c the order's largest cutoff:

        * pairs (one bond):                 R = c,  D = c       -> 2c
        * "compact" (all bonded), any n:    R = c,  D = c       -> 2c
        * "hub" (one bonded to all):        R = c,  D = 2c      -> 3c
        * "connected", n molecules: a spanning tree has a centre within
          floor(n/2) bonds of every member and a diameter of at most n - 1
          bonds:                           R = floor(n/2) c, D = (n-1) c
          -> 3c for n = 3, 5c for n = 4.

        The same bound for pairs (2c) also means a pair has at most one image
        within the cutoff. For the contact criteria the bound is on the
        reference points (centres of geometry), so each bond length becomes
        c + 2 r_max, r_max being the largest distance of an atom from its
        molecule's centre of geometry. Clusters (no cell) are always fine.
        """
        if not system.periodic:
            return
        l_min = float(system.widths.min())
        types = sorted(system.type_counts())
        if not types:
            return
        pad = 0.0
        if self.criterion in ("contact", "heavy contact"):
            pad = 2 * max(
                float(
                    np.linalg.norm(
                        m.coordinates - m.coordinates.mean(axis=0), axis=1
                    ).max()
                )
                for m in system.molecules
            )
        for n in range(2, self.max_order + 1):
            if self.rule(n) == "none":
                continue
            c = max(self.cutoff(n, a, b) for a in types for b in types)
            radius, diameter = self.reach(n)
            bound = (radius + diameter) * (c + pad)
            if not l_min > bound:
                raise SelectionError(
                    f"The order-{n} selection ({self.rule(n)}, cutoff {c:.3f} Å"
                    + (f" + {pad:.3f} Å for contact distances" if pad else "")
                    + f") needs a cell whose smallest width exceeds {bound:.3f} "
                    f"Å, but this cell's is {l_min:.3f} Å: the same molecules "
                    "could form two different fragments. Use a larger cell or "
                    "smaller cutoffs."
                )
