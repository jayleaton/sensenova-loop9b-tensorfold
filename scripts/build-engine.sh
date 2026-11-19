#!/usr/bin/env bash
# Fetch and patch one backend's engine (scripts/fetch-engine.sh), then build tensorfold-native in it.
#
#   scripts/build-engine.sh cuda DEST [SOURCE]    on a DGX Spark; NVCC (default /usr/local/cuda/bin/nvcc) builds the kernels
#   scripts/build-engine.sh cuda-v6 DEST [SOURCE] the same, with the CUDA series through 0027 (loop seam, one-barrier prefill)
#   scripts/build-engine.sh metal DEST [SOURCE]   on Apple Silicon with Xcode's macOS SDK and Metal toolchain
#   scripts/build-engine.sh rtx DEST [SOURCE]     an RTX card: the CUDA series plus EXL3, kernels for SM (default 120)
#
# NVCC=none builds the CUDA side's host code without kernel images (a source check on a machine without CUDA).
set -euo pipefail
backend=${1:?usage: scripts/build-engine.sh cuda|cuda-v6|metal|rtx DEST [SOURCE]}
dest=${2:?usage: scripts/build-engine.sh cuda|cuda-v6|metal|rtx DEST [SOURCE]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"$root/scripts/fetch-engine.sh" "$@"
want=$(cat "$dest/.zig-version")
have=$(zig version)
[[ "$have" == "$want" ]] || { echo "zig $have found; the engine needs zig $want (.zig-version)" >&2; exit 2; }
cd "$dest"
case "$backend" in
    cuda | cuda-v6)
        nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
        if [[ "$nvcc" == none ]]; then
            zig build native
        else
            zig build native -Dnvcc="$nvcc"
            zig build qwen35-cuda -Dnvcc="$nvcc"
        fi
        ;;
    metal) zig build native -Dcpu=apple_m1 ;;
    rtx)
        nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
        zig build native -Dsm="${SM:-120}" -Dnvcc="$nvcc"
        zig build qwen35-cuda -Dsm="${SM:-120}" -Dnvcc="$nvcc"
        ;;
    *) echo "no build for $backend" >&2; exit 2 ;;
esac
echo "built $dest/zig-out/native/bin/tensorfold-native"
