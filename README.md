# The Measured Orator — code and data

Reproducible analysis for *The Measured Orator: A Quantitative Aesthetics of Fatos Nano's
Prime-Ministerial Rhetoric* (Endri Prifti, 2026).

Repository: https://github.com/endriprifti/nano_measured_orator_code_data · Contact: endri@msn.com

Source: *Fjalime të z. Fatos Nano në Kuvendin e Shqipërisë (Kryeministër i vendit shkurt 1991–korrik 1991;
korrik 1997–tetor 1998; korrik 2002–shtator 2005)*. Kuvendi i Shqipërisë, Shërbimi i Botimeve Parlamentare,
Tiranë 2010. ISBN 978-99956-798-6-6.

## Run

```bash
pip install -r requirements.txt
bash run_all.sh path/to/fatos_nano.pdf [path/to/Mbledhja_e_KPD.aac [folder/with/FN1-FN3.mp4 and FN1990.m4a, FN1991.m4a]]
```

The third argument runs Appendix C.2 on the broadcast clips and the two archival audio recordings of 1990 and 1991. It needs three pretrained components that are not
redistributed here; `get_models.sh` fetches them (Silero VAD and the Resemblyzer GE2E weights from their PyPI wheels,
Whisper large-v3-turbo as an int8 ONNX export from the sherpa-onnx releases). PyTorch is not needed: the speaker
encoder runs in NumPy (`voice_encoder_np.py`) and the VAD and recogniser in onnxruntime.

The language-model step (`lm.py`) trains a 5-gram Kneser–Ney model and a 2.2M-parameter Transformer in five
cross-validation folds; it takes about 1.5 hours on two CPU cores. Every other step takes seconds to a few minutes.

## Files

| Script | Paper section | Output |
| --- | --- | --- |
| `parse.py` | 3.3 | `corpus.json`, `corpus_removed.log` — sessions, speech units, paragraphs; removed editorial and quoted material |
| `prep.py` | 3.3–3.5 | `units.json` — sentences, clauses, tokens, genres; `results/removed_counts.json` — words removed at extraction |
| `lexical.py` | 5 | `results/lexical.json` |
| `rhythm.py`, `rhythm_dfa.py` | 6 | `results/rhythm.json` |
| `figures.py` | 7 | `results/figures.json` |
| `info.py` | 8.1–8.2 | `results/info.json` |
| `lm.py` | 4.6, 8.3 | `results/surprisal.json`, `results/lm.log` |
| `surprisal.py` | 8.3–8.7 | `results/surprisal_analysis.json` |
| `semantics.py` | 9 | `results/semantics.json` |
| `stylometry.py` | 7.4, 10 | `results/stylometry.json` |
| `acoustic.py` | Appendix C.1 | `results/acoustic.json`, `results/acoustic_ipus.json` (needs ffmpeg and praat-parselmouth) |
| `video.py` | Appendix C.2 | `results/video.json`, `results/asr_cache.json`, `results/video.log` — speaker attribution, VAD-based IPUs, calibration of the syllable detector, register comparison of six recordings, alignment of clip FN2 with session S18 |
| `prosody.py`, `vad_np.py`, `voice_encoder_np.py`, `asr.py` | Appendix C.2 | shared segmentation; Silero VAD in onnxruntime; GE2E speaker encoder in NumPy; Whisper turbo via sherpa-onnx |
| `get_models.sh` | Appendix C.2 | downloads the pretrained components |
| `make_figs.py` | all figures | `figures/fig1…fig10` (PNG at 300 dpi and vector PDF) |
| `common.py` | shared | loading, figure style |

All random seeds are fixed. Numbers in the paper are read from the JSON files in `results/`.
