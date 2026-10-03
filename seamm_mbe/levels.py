"""Which low level each selected fragment's increment uses.

The cell's low level is usually a periodic plane-wave code, and the fragments'
low level need not be the same code. The prototype's *mixed* scheme references
monomers and compact pairs (O--O < 3.5 Å) to the periodic code (VASP, in
registered boxes), cancelling the cell's own low-level error best, and
everything else to a molecular code (ORCA r2SCAN-D4), because extended
fragments pick up image interactions in affordable periodic boxes.

Each increment is built entirely at its own level: a triple referenced to the
molecular level subtracts the molecular-level increments of its pairs, even
when those pairs are themselves referenced to the periodic level in the sum.
So the periodic level is computed on the periodic fragments and their
sub-fragments, and the molecular level on the molecular fragments and theirs
(:meth:`FragmentSet.calculations`).
"""

import numpy as np

PERIODIC = "periodic"
MOLECULAR = "molecular"
LEVELS = (PERIODIC, MOLECULAR)


def assign_levels(fragments, periodic=None):
    """Assign each selected fragment a low level.

    Parameters
    ----------
    fragments : seamm_mbe.FragmentSet
        The fragments; their ``level`` attributes are set.
    periodic : {int: bool or float} or None
        Per order, whether selected fragments use the periodic low level:
        True for all of that order, a distance r (Å) for those whose members
        are all closer than r (by the selection's criterion), False or absent
        for none. Only real booleans mean all or none: 1 is a distance of
        1 Å. None (the default) puts everything at the molecular level.
        The prototype's mixed scheme is ``{1: True, 2: 3.5}``.

    Returns
    -------
    {str: {int: int}}
        The number of selected fragments per level and order.
    """
    periodic = periodic or {}
    counts = {PERIODIC: {}, MOLECULAR: {}}
    for f in fragments:
        if not f.in_sum:
            f.level = None
            continue
        rule = periodic.get(f.order, False)
        if isinstance(rule, (bool, np.bool_)):
            use = bool(rule)
        elif rule is None:
            use = False
        elif isinstance(rule, (int, float, np.integer, np.floating)):
            use = f.max_distance < float(rule)
        else:
            raise ValueError(
                f"The periodic rule for order {f.order} is {rule!r}: it must be "
                "True, False or a distance in Å."
            )
        f.level = PERIODIC if use else MOLECULAR
        counts[f.level][f.order] = counts[f.level].get(f.order, 0) + 1
    return counts
