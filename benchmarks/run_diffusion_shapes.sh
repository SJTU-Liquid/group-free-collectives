#!/usr/bin/env bash
# Run gfc benchmarks on the three diffusion-serving presets.
#
# Env vars:
#   GFC_PY        path to a python that imports torch/triton (default: python3/python on PATH)
#   GFC_TORCHRUN  path to torchrun (default: torchrun on PATH)
#   GFC_GPUS      CUDA_VISIBLE_DEVICES list (default: 1,2,3,4 to avoid GPU 0)
#   GFC_NPROC     nproc_per_node (default: 4)
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# Default to whatever python / torchrun are on PATH so a fresh clone runs out
# of the box. Override either with $GFC_PY / $GFC_TORCHRUN.
py="${GFC_PY:-$(command -v python3 || command -v python || echo python)}"
torchrun="${GFC_TORCHRUN:-$(command -v torchrun || echo torchrun)}"
gpus="${GFC_GPUS:-1,2,3,4}"
nproc="${GFC_NPROC:-4}"
iters="${GFC_ITERS:-500}"
warmup="${GFC_WARMUP:-50}"

# Track validation/run failures so the harness exits non-zero instead of
# reporting green when a preset's --validate fails. The per-preset log is
# printed regardless so timings + the validation verdict are always visible.
fail=0
for preset in small_ctrl mid_act large_latent; do
    echo "=== preset=$preset nproc=$nproc ==="
    rm -rf /tmp/gfc_run
    rc=0
    CUDA_VISIBLE_DEVICES="$gpus" "$torchrun" --standalone --nproc_per_node="$nproc" \
        --redirects=3 --log-dir /tmp/gfc_run \
        benchmarks/bench_symm_collectives.py \
        --diffusion-preset "$preset" \
        --iters "$iters" --warmup "$warmup" \
        --compare-nccl-reference \
        --validate || rc=$?
    for f in /tmp/gfc_run/*/attempt_0/0/stdout.log; do
        [ -f "$f" ] && cat "$f"
    done
    if [ "$rc" -ne 0 ]; then
        echo "!!! preset=$preset FAILED (torchrun exit $rc) — validation or run error"
        fail=1
    fi
    echo
done

if [ "$fail" -ne 0 ]; then
    echo "run_diffusion_shapes: one or more presets failed"
fi
exit "$fail"
