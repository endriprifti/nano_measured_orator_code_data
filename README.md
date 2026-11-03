# The Measured Orator — code and data

Reproducible analysis for *The Measured Orator: A Quantitative Aesthetics of Fatos Nano's
Prime-Ministerial Rhetoric* (Endri Prifti, 2026).

Source: *Fjalime të z. Fatos Nano në Kuvendin e Shqipërisë (Kryeministër i vendit shkurt 1991–korrik 1991;
korrik 1997–tetor 1998; korrik 2002–shtator 2005)*. Kuvendi i Shqipërisë, Shërbimi i Botimeve Parlamentare,
Tiranë 2010. ISBN 978-99956-798-6-6.

## Run

```bash
pip install -r requirements.txt
bash run_all.sh path/to/fatos_nano.pdf [path/to/Mbledhja_e_KPD.aac]
```

The language-model step (`lm.py`) trains a 5-gram Kneser–Ney model and a 2.2M-parameter Transformer in five
cross-validation folds; it takes about 1.5 hours on two CPU cores. Every other step takes seconds to a few minutes.

## Files

| Script | Paper section | Output |
| --- | --- | --- |
| `parse.py` | 3.3 | `corpus.json`, `corpus_removed.log` — sessions, speech units, paragraphs; removed editorial and quoted material |
| `prep.py` | 3.4–3.5 | `units.json` — sentences, clauses, tokens, genres |
| `lexical.py` | 5 | `results/lexical.json` |
| `rhythm.py`, `rhythm_dfa.py` | 6 | `results/rhythm.json` |
| `figures.py` | 7 | `results/figures.json` |
| `info.py` | 8.1–8.2 | `results/info.json` |
| `lm.py` | 4.6, 8.3 | `results/surprisal.json`, `results/lm.log` |
| `surprisal.py` | 8.3–8.7 | `results/surprisal_analysis.json` |
| `semantics.py` | 9 | `results/semantics.json` |
| `stylometry.py` | 7.4, 10 | `results/stylometry.json` |
| `acoustic.py` | Appendix C | `results/acoustic.json`, `results/acoustic_ipus.json` (needs ffmpeg and praat-parselmouth) |
| `make_figs.py` | all figures | `figures/fig1…fig8` (PNG at 300 dpi and vector PDF) |
| `common.py` | shared | loading, figure style |

All random seeds are fixed. Numbers in the paper are read from the JSON files in `results/`.
