#!/usr/bin/env bash
# Reproduces every number and figure of the paper from the official PDF edition.
# Usage: bash run_all.sh path/to/fatos_nano.pdf [path/to/Mbledhja_e_KPD.aac [folder with FN1.mp4 FN2.mp4 FN3.mp4 FN5.mp4]]
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

# Appendix C (optional): acoustic pilot on the recording of 3 December 2001, Nano's segment from 2:40 to the end
if [ -n "${2:-}" ]; then
  ffmpeg -hide_banner -loglevel error -y -ss 160 -i "$2" -ac 1 -ar 16000 nano_seg.wav
  python3 acoustic.py nano_seg.wav
  python3 make_figs.py fig9
fi

# Appendix C.2 (optional): broadcast clips FN1-FN5 supplied by the author (FN4 has an empty audio track and is skipped).
# Needs the pretrained components fetched by get_models.sh (Silero VAD, GE2E speaker encoder, Whisper large-v3-turbo).
if [ -n "${3:-}" ]; then
  mkdir -p ../audio ../video
  cp nano_seg.wav ../audio/nano_seg.wav
  for c in FN1 FN2 FN3 FN5; do
    ffmpeg -hide_banner -loglevel error -y -i "$3/$c.mp4" -vn -ac 1 -ar 16000 "../video/$c.wav"
  done
  [ -f silero_vad.onnx ] || bash get_models.sh
  python3 video.py > results/video.log      # ~15 min on 2 CPU cores, mostly automatic transcription
  python3 make_figs.py fig10
fi
