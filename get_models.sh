#!/usr/bin/env bash
# Fetches the three pretrained components used in Appendix C.2 (video.py). None is redistributed with this package.
#   silero_vad.onnx            Silero VAD v5, from the PyPI wheel `silero-vad` (MIT licence)
#   resemblyzer_pretrained.pt  GE2E speaker encoder weights, from the PyPI wheel `Resemblyzer` (Apache-2.0)
#   Whisper large-v3-turbo     int8 ONNX export published by the sherpa-onnx project (k2-fsa), MIT licence (Whisper)
set -euo pipefail
TMP=$(mktemp -d)
pip download --no-deps -d "$TMP" silero-vad==6.2.3 Resemblyzer==0.1.4 >/dev/null
( cd "$TMP" && unzip -oq silero_vad-6.2.3-py3-none-any.whl 'silero_vad/data/silero_vad.onnx' \
             && unzip -oq Resemblyzer-0.1.4-py3-none-any.whl 'resemblyzer/pretrained.pt' )
cp "$TMP/silero_vad/data/silero_vad.onnx" silero_vad.onnx
cp "$TMP/resemblyzer/pretrained.pt" resemblyzer_pretrained.pt
mkdir -p ../models && cd ../models
curl -sSL -o turbo.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2
tar xjf turbo.tar.bz2 && rm turbo.tar.bz2
rm -rf "$TMP"
echo "models ready: silero_vad.onnx, resemblyzer_pretrained.pt, ../models/sherpa-onnx-whisper-turbo/"
