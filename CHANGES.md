# Changes after the review of 4 November 2026

Measurement
- `figures.py`: second-person-plural verbs no longer count nouns in -ioni and eight other nouns in -oni;
  address by name excludes titles of office (Kryetar, President) and matches sentence-initial "Zoti";
  Benjamini–Hochberg p-values over all 26 markers saved as `p_bh`; anaphora without titles of office and
  without "Në", and the most frequent shared endings, are now computed (separate random stream).
- `stylometry.py`: negative-binomial (NB2) models estimate the dispersion by maximum likelihood
  (`smf.negativebinomial`, exposure = words) instead of fixing alpha = 0.3.
- `surprisal.py`: sentence-level analyses use the same sentences as Sections 6–7 (vocative address lines
  excluded) and the clause segmentation of `prep.py`; model-level rates still use every scored word.
- `semantics.py`: ring composition is tested against non-overlapping segment pairs at least half a speech
  apart (the unmatched null is kept as `ring_pct_unmatched`); random coherence pairs exclude self-pairs.
- `rhythm.py`: Weibull parameters, empirical closure-hazard bands and the arc without exclamations are saved.
- `info.py`: most frequent two-word sentence openings per premiership are saved.
- `prep.py`: words removed at extraction and residual roman-type questions are counted
  (`results/removed_counts.json`).
- `video.py`: corrections to Appendix C.1 (calibrated rates, corrected KS distance) and pitch intervals over
  IPUs are saved.

Figures
- Figure 3a shows clause epistrophe against both nulls; Figure 4 and Figure 6 (surprisal) are redrawn;
  Figures 5 and 6 are renumbered to follow their order of citation (`fig5_order_complexity`, `fig6_surprisal`);
  Figure 10e uses pitch medians over IPUs.
