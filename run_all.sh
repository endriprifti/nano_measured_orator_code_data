#!/usr/bin/env bash
# Reproduces every number and figure of the paper from the official PDF edition.
# Usage: bash run_all.sh path/to/fatos_nano.pdf [path/to/Mbledhja_e_KPD.aac [folder with FN1.mp4 FN2.mp4 FN3.mp4 FN1990.m4a FN1991.m4a]]
set -euo pipefail
PDF="${1:?give the path to the PDF of 'Fjalime të z. Fatos Nano në Kuvendin e Shqipërisë' (Tiranë 2010)}"
python3 parse.py "$PDF" corpus.json        # typography-based segmentation (sessions, speech units, paragraphs)
python3 prep.py                            # tokens, sentences, clauses, genres -> units.json
python3 lexical.py                         # Section 5
python3 rhythm.py                          # Section 6
python3 figures.py                         # Section 7
python3 info.py                            # Section 8 (model-free)
python3 lm.py kn gpt                       # Section 8 (cross-validated KN 5-gram and Transformer; ~1.5 h on 2 CPU cores)
python3 surprisal.py                       # Section 8 (surprisal analyses)
python3 semantics.py                       # Section 9
python3 stylometry.py                      # Sections 7.4 and 10
python3 make_figs.py fig1 fig2 fig3 fig4 fig5 fig6 fig7 fig8

# Section 11: comparison with Marx's German originals (texts fetched from a pinned public mirror, not redistributed)
bash get_marx.sh
python3 marx_corpus.py > results/marx_corpus.log   # Marx corpus -> marx_corpus.json (local), results/marx_corpus_stats.json
python3 marx.py > results/marx.log                 # ~10 min on 2 CPU cores
python3 make_figs.py fig_marx

# Appendix C (optional): acoustic pilot on the recording of 3 December 2001, Nano's segment from 2:40 to the end
if [ -n "${2:-}" ]; then
  ffmpeg -hide_banner -loglevel error -y -ss 160 -i "$2" -ac 1 -ar 16000 nano_seg.wav
  python3 acoustic.py nano_seg.wav
  python3 make_figs.py fig9              # Figure 10 of the paper
fi

# Appendix C.2 (optional): broadcast clips FN1-FN3 and archival audio FN1990, FN1991 supplied by the author
# (FN4 has an empty audio track and is skipped; FN5 was withdrawn).
# Needs the pretrained components fetched by get_models.sh (Silero VAD, GE2E speaker encoder, Whisper large-v3-turbo).
if [ -n "${3:-}" ]; then
  mkdir -p ../audio ../video ../audio90
  cp nano_seg.wav ../audio/nano_seg.wav
  for c in FN1 FN2 FN3; do
    ffmpeg -hide_banner -loglevel error -y -i "$3/$c.mp4" -vn -ac 1 -ar 16000 "../video/$c.wav"
  done
  for c in FN1990 FN1991; do
    ffmpeg -hide_banner -loglevel error -y -i "$3/$c.m4a" -vn -ac 1 -ar 16000 "../audio90/$c.wav"
  done
  [ -f silero_vad.onnx ] || bash get_models.sh
  python3 video.py > results/video.log      # ~30 min on 2 CPU cores, mostly automatic transcription
  python3 make_figs.py fig10             # Figure 11 of the paper
fi
