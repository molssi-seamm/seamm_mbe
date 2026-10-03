"""Element symbols, atomic numbers and standard atomic weights.

The weights are the IUPAC abridged standard atomic weights (conventional values
for elements with an interval), used only for molecular centres of mass. They
are kept here, for H to Ba, so that common systems need nothing beyond numpy;
heavier elements fall back to molsystem's table.
"""

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn"
    " Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La"
    " Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po"
    " At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg"
    " Cn Nh Fl Mc Lv Ts Og"
).split()

ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(SYMBOLS, start=1)}

_WEIGHTS = (
    "1.008 4.0026 6.94 9.0122 10.81 12.011 14.007 15.999 18.998 20.180 22.990 24.305"
    " 26.982 28.085 30.974 32.06 35.45 39.95 39.098 40.078 44.956 47.867 50.942"
    " 51.996 54.938 55.845 58.933 58.693 63.546 65.38 69.723 72.630 74.922 78.971"
    " 79.904 83.798 85.468 87.62 88.906 91.224 92.906 95.95 97.907 101.07 102.91"
    " 106.42 107.87 112.41 114.82 118.71 121.76 127.60 126.90 131.29 132.91 137.33"
).split()

MASS = {symbol: float(w) for symbol, w in zip(SYMBOLS, _WEIGHTS)}

#: Elements that are ions in molecular systems: never bonded (as molsystem)
IONIC_ELEMENTS = frozenset("Li Na K Rb Cs Be Mg Ca Sr Ba".split())

#: Elements that can be a molecule on their own: those ions, the halide ions
#: and the noble gases
MONATOMIC = IONIC_ELEMENTS | frozenset("F Cl Br I He Ne Ar Kr Xe Rn".split())


def atomic_number(symbol):
    """The atomic number of an element symbol."""
    try:
        return ATOMIC_NUMBER[symbol]
    except KeyError:
        raise ValueError(f"Unknown element symbol {symbol!r}") from None


def mass(symbol):
    """The standard atomic weight (g/mol) of an element symbol.

    The table here (H--Ba) uses IUPAC's conventional values, as the prototype
    did (O 15.999); heavier elements come from molsystem's table.
    """
    if symbol in MASS:
        return MASS[symbol]
    from molsystem.elements import symbol_to_mass

    try:
        return float(symbol_to_mass[symbol])
    except KeyError:
        raise ValueError(f"No atomic weight for {symbol!r}") from None


def hill_formula(symbols):
    """The Hill-order formula, e.g. 'H2O', 'C3H4O3', 'BF4', 'Li'."""
    counts = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else [])
        order += sorted(s for s in counts if s not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order)
