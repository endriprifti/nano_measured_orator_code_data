# Changes of 5 November 2026 (afternoon): recogniser fix

Found with the author's exact transcriptions, by ear, of one window of each tape.
- `asr.py`: every stretch is decoded with 5 s of silence appended. Without it the decoder dropped the last one to two
  seconds of speech (in a 24.7-s window of FN1990 it lost the final 30 of 163 syllables). Stretches that would not fit
  one padded Whisper window are cut at the longest pause in their middle third (`video.split_long`).
- `video.py`: of the three decodes with widened edges, the longest that does not loop is kept (was: the median),
  because the decoder's failures on these recordings are omissions; on the three exact transcripts the two rules agree
  within 1%. A calibration window whose detector/transcript ratio exceeds 1.5 times the recording's median counts as
  stopped early. The bias of the automatic syllable count is pooled over three exact transcripts (FN2 printed text and
  the two tape windows, `EXACT_TAPES`): 0.967.
- `video.py`: the IPU-to-text alignment of FN2 scores matched characters minus half the unmatched ones. The previous
  score, proportional to the longer string, attached "tashmë" to IPU 6 although both decodes of IPU 5 contain it;
  Table C2 now reads IPU 5 "lejon tashmë" (3 syllables) and IPU 6 from "çrrënjosjen" (17).
- All ASR-based numbers were recomputed (`results/asr_cache.json` rebuilt; keys now carry the padding).

# Changes of 5 November 2026

Appendix C.2
- The 29-s video clip of a 1991 address (FN5) is withdrawn at the author's request; it is no longer analysed and no
  longer enters the calibration. The bias of the automatic syllable count now rests on the exact transcript of FN2
  alone (`asr_bias`, previously `asr_bias_pooled` over FN2 and FN5), which shifts the calibrated rates slightly.
- Two archival audio recordings supplied by the author are added: FN1990 (speech at a party aktiv, 1990) and FN1991
  (Nano as Chairman of the Council of Ministers at a pre-election meeting of cadres, March 1991). The chair's
  introduction is located in the automatic transcript; the remainder is tested for a second voice. Syllables are
  counted with the nucleus detector calibrated on 12 transcribed windows of each recording, as for the KPD recording.
- `video.py` reads the archival WAVs from `../audio90` (`NANO_AUDIO90`); speaker embeddings are computed in batches.
- Figure 10b–e shows six recordings.
- `results/video.json` also records, per recording, the speech rate including silent pauses (`speech_rate_est`)
  and the share of speaking time filled by phonation (`speech_share`), used to compare tempo before and after 1997.

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
