#!/usr/bin/env bash
# Section 11: fetches Marx's texts from a public GitHub mirror, pinned to the commit used in the paper:
#   mewerke/   HTML of the Marx-Engels-Werke (MEW, Dietz Verlag, Berlin), German originals
#   en/        English translations of four of the works (translation check only)
# The texts are not redistributed with this package. Destination: $MARX_MLRAW (default ../marx-ebooks/mlraw).
set -euo pipefail
DEST=${MARX_MLRAW:-../marx-ebooks/mlraw}
COMMIT=a3a7d3ea087e9ca069f10f5101103c8abe8afb22
mkdir -p "$DEST" && cd "$DEST"
[ -d .git ] || { git init -q && git remote add origin https://github.com/MARX-EBOOKS/mlraw; }
git sparse-checkout init --no-cone
git sparse-checkout set 'mewerke/' 'en/Manifesto_of_the_Communist_Party/' \
  '/en/The_Eighteenth_Brumaire_of_Louis_Bonaparte.html' '/en/The_Class_Struggles_in_France,_1848_to_1850.html' \
  '/en/Critique_of_the_Gotha_Programme.html'
GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git checkout -q FETCH_HEAD
echo "Marx texts ready in $DEST (commit $COMMIT)"
