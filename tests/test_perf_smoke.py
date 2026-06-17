"""Phase-8 perf smoke test.

Compares gfc all_gather and all2all against ``dist.all_gather_into_tensor``
and ``dist.all_to_all_single`` at small / medium / large sizes.
Reference-only — does not assert on timings (variance is large on a busy
node).

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_perf_smoke.py
    torchrun --standalone --nproc_per_node=4 tests/test_perf_smoke.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)

from tests._harness import rank0_print, setup_nccl, teardown  # noqa: E402


def _time_gpu(fn: Callable[[], None], n_iter: int, n_warmup: int, stream) -> float:
    """Median per-iter latency in microseconds."""
    for _ in range(n_warmup):
        fn()
    stream.synchronize()
    samples = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert world in (2, 4)

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=16 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g = runtime.register_group(tuple(range(world)))

    n_iter = 200
    n_warmup = 20

    rank0_print(f"\n=== perf_smoke (world={world}) ===")
    rank0_print(
        f"{'collective':<12} {'bytes':>10} {'gfc us':>10} {'nccl us':>10} {'gfc/nccl':>9}"
    )

    # all_gather
    for nbytes in (1024, 64 * 1024, 1 * 1024 * 1024):
        inp = torch.empty(nbytes, dtype=torch.uint8, device=device)
        inp.fill_(runtime.rank)
        out = torch.empty(world * nbytes, dtype=torch.uint8, device=device)

        with torch.cuda.stream(runtime.stream):
            gfc_us = _time_gpu(
                lambda: runtime.all_gather(inp, out, g),
                n_iter, n_warmup, runtime.stream,
            )

        ref_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(ref_stream):
            nccl_us = _time_gpu(
                lambda: dist.all_gather_into_tensor(out, inp),
                n_iter, n_warmup, ref_stream,
            )

        rank0_print(
            f"{'all_gather':<12} {nbytes:>10} {gfc_us:>10.1f} {nccl_us:>10.1f} {gfc_us/nccl_us:>9.2f}"
        )

    # all2all
    for slice_bytes in (1024, 64 * 1024, 1 * 1024 * 1024 // world):
        if slice_bytes * world > config.max_collective_bytes:
            continue
        inp = torch.empty(world * slice_bytes, dtype=torch.uint8, device=device)
        inp.fill_(runtime.rank)
        out = torch.empty(world * slice_bytes, dtype=torch.uint8, device=device)

        with torch.cuda.stream(runtime.stream):
            gfc_us = _time_gpu(
                lambda: runtime.all2all(inp, out, g),
                n_iter, n_warmup, runtime.stream,
            )

        ref_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(ref_stream):
            nccl_us = _time_gpu(
                lambda: dist.all_to_all_single(out, inp),
                n_iter, n_warmup, ref_stream,
            )

        total = slice_bytes * world
        rank0_print(
            f"{'all2all':<12} {total:>10} {gfc_us:>10.1f} {nccl_us:>10.1f} {gfc_us/nccl_us:>9.2f}"
        )

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
