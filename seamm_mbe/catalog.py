"""Molecule types: identification by formula and bond-graph topology.

A molecule's *signature* is its Hill formula plus a Weisfeiler--Lehman hash of
its element-labelled bond graph, so isomers with the same formula are told
apart (in practice; WL refinement does not separate every pair of
non-isomorphic graphs, which is irrelevant for the small molecules here). The
same refinement labels each atom by its symmetry class within the molecule,
which is how a type's *designated atom* (the atom whose position defines the
molecule's position for the distance criterion: water's O, a carbonate's
carbonyl C, an ion's central atom) is found in every molecule of that type.

The built-in :data:`CATALOG` knows the molecules of the water/electrolyte
campaign: water, ethylene carbonate (EC), fluoroethylene carbonate (FEC),
dimethyl carbonate (DMC), ethyl methyl carbonate (EMC), Li⁺, BF₄⁻ and PF₆⁻,
plus the common monatomic ions Na⁺, K⁺, F⁻, Cl⁻, Br⁻ and I⁻.
Anything else is typed automatically, named by its formula.
"""

from dataclasses import dataclass
import hashlib


def _digest(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def wl_labels(symbols, bonds):
    """The Weisfeiler--Lehman atom labels of an element-labelled graph.

    Parameters
    ----------
    symbols : [str]
        Element symbols of the molecule's atoms.
    bonds : [(int, int)]
        Bonds as pairs of 0-based indices into ``symbols``.

    Returns
    -------
    [str]
        A label per atom; atoms with equal labels are (WL-)equivalent.
    """
    n = len(symbols)
    neighbors = [[] for _ in range(n)]
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    labels = list(symbols)
    n_classes = len(set(labels))
    for _ in range(n):
        labels = [
            _digest(labels[i] + "(" + ",".join(sorted(labels[j] for j in nbrs)) + ")")
            for i, nbrs in enumerate(neighbors)
        ]
        if len(set(labels)) == n_classes:
            break
        n_classes = len(set(labels))
    return labels


def signature(symbols, bonds):
    """The topology signature of a molecule: formula + WL graph hash."""
    from .elements import hill_formula

    labels = wl_labels(symbols, bonds)
    return hill_formula(symbols) + ":" + _digest(",".join(sorted(labels)))


@dataclass
class MoleculeType:
    """A kind of molecule.

    Attributes
    ----------
    name : str
        The type's name, e.g. "water", "EC", "Li+", or the formula for an
        automatically typed molecule.
    formula : str
        Hill formula.
    signature : str
        Formula + WL graph hash (see :func:`signature`).
    charge : int or None
        The molecule's total charge; None if the type does not fix it.
    multiplicity : int
        Spin multiplicity, 2S + 1.
    designated : str or None
        The WL label of the designated atom, or None to use the heavy atom
        nearest the centre of mass (set when the type is first seen).
    """

    name: str
    formula: str
    signature: str
    charge: int | None = 0
    multiplicity: int = 1
    designated: str | None = None


def define_type(name, symbols, bonds, charge=0, multiplicity=1, designated=None):
    """A :class:`MoleculeType` from a template graph.

    Parameters
    ----------
    name : str
        The type's name.
    symbols : [str]
        Element symbols of the template.
    bonds : [(int, int)]
        The template's bonds (0-based).
    charge, multiplicity : int
        Defaults for the type.
    designated : int or None
        Index of the designated atom in the template.
    """
    from .elements import hill_formula

    labels = wl_labels(symbols, bonds)
    return MoleculeType(
        name=name,
        formula=hill_formula(symbols),
        signature=signature(symbols, bonds),
        charge=charge,
        multiplicity=multiplicity,
        designated=None if designated is None else labels[designated],
    )


def _carbonate(name, substituents, ring=()):
    """A cyclic or linear carbonate template. Atoms 0-3 are the carbonyl C, the
    carbonyl O and the two ester O; ``substituents`` adds the rest as
    (symbol, bonded-to) pairs, indices counting on from 4, and ``ring`` any
    further bonds that close a ring."""
    symbols = ["C", "O", "O", "O"]
    bonds = [(0, 1), (0, 2), (0, 3)]
    for symbol, partner in substituents:
        bonds.append((partner, len(symbols)))
        symbols.append(symbol)
    return define_type(name, symbols, bonds + list(ring), designated=0)


#: Ethylene carbonate: ring O2-C4-C5-O3 (closed by the O3-C5 ring bond)
_EC = [("C", 2), ("C", 4), ("H", 4), ("H", 4), ("H", 5), ("H", 5)]
#: Fluoroethylene carbonate: EC with F on C4
_FEC = [("C", 2), ("C", 4), ("F", 4), ("H", 4), ("H", 5), ("H", 5)]
#: Dimethyl carbonate: CH3 on each ester O
_DMC = [("C", 2), ("C", 3), ("H", 4), ("H", 4), ("H", 4), ("H", 5), ("H", 5), ("H", 5)]
#: Ethyl methyl carbonate: CH3 on O2, CH2-CH3 on O3
_EMC = [("C", 2), ("C", 3), ("C", 5)] + [("H", 4)] * 3 + [("H", 5)] * 2 + [("H", 6)] * 3


def _catalog():
    types = [
        define_type("water", ["O", "H", "H"], [(0, 1), (0, 2)], designated=0),
        _carbonate("EC", _EC, ring=[(3, 5)]),
        _carbonate("FEC", _FEC, ring=[(3, 5)]),
        _carbonate("DMC", _DMC),
        _carbonate("EMC", _EMC),
        define_type("Li+", ["Li"], [], charge=1, designated=0),
        define_type("Na+", ["Na"], [], charge=1, designated=0),
        define_type("K+", ["K"], [], charge=1, designated=0),
        define_type("F-", ["F"], [], charge=-1, designated=0),
        define_type("Cl-", ["Cl"], [], charge=-1, designated=0),
        define_type("Br-", ["Br"], [], charge=-1, designated=0),
        define_type("I-", ["I"], [], charge=-1, designated=0),
        define_type("BF4-", ["B"] + ["F"] * 4, [(0, k) for k in range(1, 5)], -1, 1, 0),
        define_type("PF6-", ["P"] + ["F"] * 6, [(0, k) for k in range(1, 7)], -1, 1, 0),
    ]
    return {t.signature: t for t in types}


#: The built-in molecule types, keyed by signature
CATALOG = _catalog()
