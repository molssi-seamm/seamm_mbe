"""Many-body expansion (MBE) corrections for periodic cells and clusters.

The energy, forces and stress of a system at a high level of theory are
estimated as a cheap calculation on the whole system plus many-body increments
of [high - low] computed on small isolated fragments (monomers, selected pairs
and triples, ...). This library holds the bookkeeping, with no calculations
and no SEAMM GUI: molecule typing, fragment enumeration with periodic images
and canonical keys, the mixed low-level assignment, the increment algebra for
energy, forces and virial, and the assembled labels. The ``mbe_step`` plug-in
runs the calculations. See ``docs/developer_guide/campaigns/2026-10-03/``.
"""

from .algebra import (  # noqa: F401
    Correction,
    FragmentResult,
    Increment,
    MissingFragmentsError,
    increments,
    mbe_correction,
    missing_results,
)
from .assemble import (  # noqa: F401
    CellTerm,
    Labels,
    assemble,
    energy_offset,
    intramolecular_virial,
)
from .catalog import CATALOG, MoleculeType, define_type, signature  # noqa: F401
from .fragments import (  # noqa: F401
    Fragment,
    FragmentSet,
    canonical_key,
    enumerate_fragments,
    low_level_of,
)
from .levels import LEVELS, MOLECULAR, PERIODIC, assign_levels  # noqa: F401
from .selection import (  # noqa: F401
    CRITERIA,
    RULES,
    SelectionError,
    SelectionRules,
)
from .system import Molecule, StructureError, System  # noqa: F401
from . import units  # noqa: F401
from ._version import __version__  # noqa: F401
