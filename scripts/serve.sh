#!/usr/bin/env bash
# Serve a converted SenseNova-Loop-9B tree with one backend's measured profile (config/<backend>/serve[-PROFILE].args).
#
#   scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [extra tensorfold-native serve flags]
#
# PORT (default 8090) and NAME (default loop9b) set the listen port and the served model name; PROFILE=vision picks
# config/<backend>/serve-vision.args (Metal: image input). A profile's .env file beside it (KEY=VALUE lines) is exported
# first; a variable already set in the environment wins.
set -euo pipefail
backend=${1:?usage: scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [flags]}
engine=${2:?usage: scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [flags]}
model=${3:?usage: scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [flags]}
shift 3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ "$backend" =~ ^[a-z0-9-]+$ && -d "$root/config/$backend" ]] || { echo "no backend config/$backend" >&2; exit 2; }
args=$root/config/$backend/serve${PROFILE:+-$PROFILE}.args
[[ -f "$args" ]] || { echo "no profile $args" >&2; exit 2; }
read -r -a profile < "$args"
env_file=${args%.args}.env
if [[ -f "$env_file" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "bad line in $env_file: $key" >&2; exit 2; }
        [[ -n "${!key+x}" ]] || export "$key=$value"
    done < "$env_file"
fi
exec "$engine/zig-out/native/bin/tensorfold-native" serve "$model" --name "${NAME:-loop9b}" --port "${PORT:-8090}" \
    "${profile[@]}" "$@"
