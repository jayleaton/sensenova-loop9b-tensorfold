#!/usr/bin/env bash
# Serve a converted SenseNova-Loop-9B tree with one backend's measured profile (config/<backend>/serve.args).
#
#   scripts/serve.sh cuda|metal ENGINE_DIR MODEL_DIR [extra tensorfold-native serve flags]
#
# PORT (default 8090) and NAME (default loop9b) set the listen port and the served model name.
set -euo pipefail
backend=${1:?usage: scripts/serve.sh cuda|metal ENGINE_DIR MODEL_DIR [flags]}
engine=${2:?usage: scripts/serve.sh cuda|metal ENGINE_DIR MODEL_DIR [flags]}
model=${3:?usage: scripts/serve.sh cuda|metal ENGINE_DIR MODEL_DIR [flags]}
shift 3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$backend" in
    cuda | metal) ;;
    *) echo 'backend must be cuda or metal' >&2; exit 2 ;;
esac
read -r -a profile < "$root/config/$backend/serve.args"
exec "$engine/zig-out/native/bin/tensorfold-native" serve "$model" --name "${NAME:-loop9b}" --port "${PORT:-8090}" \
    "${profile[@]}" "$@"
