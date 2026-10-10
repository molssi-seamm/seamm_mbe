=======
History
=======
2026.10.10 -- Ion shells: an ion and its first shell as one unit
    * ``ion_shells(system, IonShellRules(...))`` returns a ``UnitSystem``: the
      system seen as units, each Li⁺ with the molecules in its first shell as one
      unit (whole around the ion) and every other molecule by itself, rebuilt from
      each geometry. ``enumerate_fragments`` builds the fragments from it unchanged.
      Around Li⁺ the per-molecule expansion converges slowly; with the shell as one
      unit a test cluster's error at triples fell from -11 to +2 kJ/mol and its
      forces from 16 to 3.6 meV/Å RMS.
    * A molecule joins a shell when one of its atoms of a listed element is within
      the cutoff of the ion (default Li-O and Li-F 2.6 Å). One within reach of two
      ions joins the nearer one, and ``max_members`` (default 5) keeps a crowded
      shell's nearest molecules. A contact anion joins the cation's shell.
    * ``SelectionRules.shell_max_order`` selects a shell's fragments only up to that
      order, e.g. 2 for its pairs but not its triples.
    * Shells need a contact criterion: a shell's reference point is its ion, so
      ``enumerate_fragments`` refuses the point criteria with shells.
    * The neighbour search now looks as many layers of periodic images out as the
      cutoffs need (at most 3), so a cell narrower than the cutoff plus the largest
      molecule's diameter is no longer refused. Large units such as Li⁺ shells
      (radius up to 7 Å) made cells of about 17.6-18.1 Å, e.g. 1 m LiPF₆ or LiBF₄ in
      EC:DMC with 4 salt pairs, fail that bound in 2-10% of frames. A molecule within
      the pair cutoff of its own image is still refused. Ordinary cells give exactly
      the same fragments as before.

2026.10.6.1 -- The molecular pressure by body order
    * ``Labels.per_body`` and ``Labels.breakdown`` give each body order's, each cell
      term's and the correction's share of the molecular pressure (``"molecular
      pressure"``) as well as of the atomic one. Each is its virial less its own
      intramolecular part, so they add up to the totals, and the monomers' share is
      zero. ``Correction.per_body`` now carries each order's forces.
2026.10.6 -- A low level per order
    * ``mbe_correction`` takes ``low_by_order``, e.g. ``{3: results}``, so the triples'
      increments can use their own molecular low level, such as a smaller basis than the
      pairs'. Each order's increments are built entirely at that level: a triple, its
      pairs and its monomers. ``FragmentSet.calculations(low_levels=...)`` lists what
      each level must compute, and each increment records the low level it used.
    * With the same results for both levels the correction is exactly the single-level
      one.
2026.10.5.1 -- Bugfix: fragments through different images in small cells
    * In a cell narrower than the cutoffs need for uniqueness, the same molecules
      can form several different triples through different images. These were
      refused; now each is enumerated and computed, named with its images, and
      ``SelectionRules.check`` warns instead of refusing.
    * Cells too small for any correct fragment list are still refused, with the
      reason: the neighbour search would miss partners, or a molecule is within the
      pair cutoff of its own image.
2026.10.5 -- A high level per order
    * ``mbe_correction`` takes ``high_by_order``, e.g. ``{3: results}``, so the triples'
      increments can use their own high level, such as a smaller basis than the pairs'.
      Each order's increments are built entirely at that level: a triple, its pairs and
      its monomers.
    * ``FragmentSet.calculations(high_levels=...)`` lists what each level must
      compute, and each increment records the high level it used. With the same
      results for both levels the correction is exactly the single-level one.
2026.10.3.1 -- Corrections to selected increments, for pairwise counterpoise
    * ``mbe_correction`` takes ``corrections``, added to selected fragments' increments
      in the sum only, never to the sub-fragment increments that higher fragments
      subtract. A pairwise counterpoise correction enters this way, so a pair's
      basis-set superposition error does not move into the 3-body terms.
2026.10.3 -- Initial release: many-body expansion (MBE) corrections for periodic cells
    * A library for the bookkeeping of many-body corrections, which estimate high-level
      energies, forces and stress for a periodic cell or a large cluster as a cheap
      calculation of the whole system plus [high - low] increments computed on small
      isolated fragments. It runs no calculations; the mbe_step plug-in will.
    * Molecules are found from the bonds, made whole across the cell, and typed by
      formula and topology, with charges from the structure's formal charges or a
      catalog: water, EC, FEC, DMC, EMC, Li+, BF4-, PF6- and common monatomic ions.
    * Fragments (monomers, pairs and triples, general in the order) are selected by a
      distance criterion with cutoffs that may depend on the pair of molecule types,
      and a rule for triples and higher (connected, hub or compact). A cell too small
      for the cutoffs is refused rather than giving duplicate fragments.
    * Each increment can use a periodic or a molecular low level (the "mixed"
      scheme); the energy, forces and an origin-independent virial are assembled into
      labels with per-type energy offsets, atomic and molecular pressure, a per-body
      breakdown and quality checks. Missing fragment results are reported, never
      silently skipped.
    * Reproduces the prototype that labelled 152 periodic water cells, from its stored
      fragment results.
