#!/usr/bin/env bash
# Run the full gfc test suite. Each multi-process test is launched via
# torchrun. GPU set is selected with $GFC_GPUS (default 1,2,3,4) to avoid
# binding GPU 0 if it is already in use.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# Default to whatever python / torchrun are on PATH so a fresh clone runs out
# of the box. Override either with $GFC_PY / $GFC_TORCHRUN.
py="${GFC_PY:-$(command -v python3 || command -v python || echo python)}"
torchrun="${GFC_TORCHRUN:-$(command -v torchrun || echo torchrun)}"
gpus="${GFC_GPUS:-1,2,3,4}"
cmd_timeout="${GFC_TEST_TIMEOUT:-180s}"

# Number of comma-separated GPUs available.
nallgpus=$(awk -F, '{print NF}' <<<"$gpus")

run_single () {
    echo "=== pytest $1 ==="
    "$py" -m pytest "$1" -q
}

run_torchrun () {
    local nproc="$1" path="$2"
    if [ "$nproc" -gt "$nallgpus" ]; then
        echo "=== SKIP $path (need $nproc GPUs, have $nallgpus from \$GFC_GPUS) ==="
        return 0
    fi
    echo "=== torchrun --nproc_per_node=$nproc $path ==="
    rm -rf /tmp/gfc_run
    CUDA_VISIBLE_DEVICES="$gpus" timeout "$cmd_timeout" "$torchrun" \
        --standalone --nproc_per_node="$nproc" \
        --redirects=3 --log-dir /tmp/gfc_run "$path"
}

run_torchrun_copy_engine () {
    local nproc="$1" path="$2"
    if [ "$nproc" -gt "$nallgpus" ]; then
        echo "=== SKIP copy-engine $path (need $nproc GPUs, have $nallgpus from \$GFC_GPUS) ==="
        return 0
    fi
    echo "=== SYMM_COLL_USE_COPY_ENGINE=1 torchrun --nproc_per_node=$nproc $path ==="
    rm -rf /tmp/gfc_run
    CUDA_VISIBLE_DEVICES="$gpus" SYMM_COLL_USE_COPY_ENGINE=1 timeout "$cmd_timeout" "$torchrun" \
        --standalone --nproc_per_node="$nproc" \
        --redirects=3 --log-dir /tmp/gfc_run "$path"
}

# Phase 1
run_single tests/test_skeleton_unit.py

# Phase 2
run_torchrun 2 tests/test_init.py
run_torchrun 4 tests/test_init.py

# Phase 3
run_torchrun 2 tests/test_remote_copy_kernel.py

# Phase 4
run_torchrun 2 tests/test_barrier_repeated.py
run_torchrun 4 tests/test_subgroup_barrier.py
run_torchrun 4 tests/test_barrier_stress.py
run_torchrun 8 tests/test_barrier_stress.py

# Phase 5
run_torchrun 2 tests/test_all_gather.py
run_torchrun 4 tests/test_all_gather.py

# Phase 6
run_torchrun 2 tests/test_all2all.py
run_torchrun 4 tests/test_all2all.py

# Phase 7
run_torchrun 2 tests/test_p2p.py

# Copy-engine correctness coverage.
run_torchrun_copy_engine 2 tests/test_all_gather.py
run_torchrun_copy_engine 2 tests/test_all2all.py
run_torchrun_copy_engine 2 tests/test_p2p.py

# Phase 8
run_torchrun 2 tests/test_cross_kernel_publication.py
run_torchrun 3 tests/test_overlap_order.py
run_torchrun 4 tests/test_complex_groups.py
run_torchrun 8 tests/test_complex_groups.py
run_torchrun 2 tests/test_buffer_reuse.py
run_torchrun 2 tests/test_perf_smoke.py
run_torchrun 4 tests/test_perf_smoke.py

# Dynamic per-group rank-list allocation (group lifecycle: register/unregister,
# >64 groups, re-register with monotonic epochs).
run_torchrun 2 tests/test_dynamic_group_alloc.py
run_torchrun 8 tests/test_dynamic_group_alloc.py

# Code-review fix regressions (arg validation, pipelined alignment, autotune
# guards, fused overlapping-group reuse).
run_torchrun 2 tests/test_review_fixes.py
run_torchrun 4 tests/test_review_fixes.py

# Phase 10
run_torchrun 2 tests/test_tma_probe.py

# Phase 12 perf (informational)
run_torchrun 4 tests/test_perf_smoke_tma.py
