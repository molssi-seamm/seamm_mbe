=======
History
=======
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
