"""Perf smoke with TMA enabled (compare against vec and NCCL)."""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import SymmetricCollectiveConfig, SymmetricCollectiveRuntime  # noqa: E402
from gfc.tma_probe import TMAUnsupportedError  # noqa: E402

from tests._harness import rank0_print, setup_nccl, teardown  # noqa: E402


def _time_gpu(fn: Callable[[], None], n_iter: int, n_warmup: int, stream) -> float:
    for _ in range(n_warmup):
        fn()
    stream.synchronize()
    samples: List[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()

    n_iter = 200
    n_warmup = 20
    sizes = (4 * 1024, 64 * 1024, 1 * 1024 * 1024, 4 * 1024 * 1024)

    rank0_print(f"\n=== perf_smoke_tma (world={world}) ===")
    rank0_print(f"{'collective':<12} {'bytes':>10} {'vec us':>10} {'tma us':>10} {'nccl us':>10} {'tma/nccl':>9}")

    for use_tma in (False, True):
        config = SymmetricCollectiveConfig(
                        max_group_size=world,
            max_collective_bytes=64 * 1024 * 1024,
            use_tma=use_tma,
        )
        try:
            runtime = SymmetricCollectiveRuntime(config, device=device)
        except TMAUnsupportedError as e:
            if use_tma:
                rank0_print(f"TMA perf smoke skipped: {e}")
                tma_results = {}
                continue
            raise
        g = runtime.register_group(tuple(range(world)))

        results = {}
        for nbytes in sizes:
            inp = torch.empty(nbytes, dtype=torch.uint8, device=device)
            inp.fill_(runtime.rank)
            out = torch.empty(world * nbytes, dtype=torch.uint8, device=device)
            with torch.cuda.stream(runtime.stream):
                ag = _time_gpu(lambda: runtime.all_gather(inp, out, g), n_iter, n_warmup, runtime.stream)
            results[("all_gather", nbytes)] = ag

        for total in sizes:
            slice_bytes = total // world
            if slice_bytes == 0:
                continue
            inp = torch.empty(total, dtype=torch.uint8, device=device)
            inp.fill_(runtime.rank)
            out = torch.empty(total, dtype=torch.uint8, device=device)
            with torch.cuda.stream(runtime.stream):
                a2a = _time_gpu(lambda: runtime.all2all(inp, out, g), n_iter, n_warmup, runtime.stream)
            results[("all2all", total)] = a2a

        # cache
        if use_tma:
            tma_results = results
        else:
            vec_results = results

        runtime.shutdown()

    # NCCL reference
    nccl_results = {}
    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=64 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g = runtime.register_group(tuple(range(world)))
    ref_stream = torch.cuda.Stream(device=device)
    for nbytes in sizes:
        inp = torch.empty(nbytes, dtype=torch.uint8, device=device)
        out = torch.empty(world * nbytes, dtype=torch.uint8, device=device)
        with torch.cuda.stream(ref_stream):
            n = _time_gpu(lambda: dist.all_gather_into_tensor(out, inp), n_iter, n_warmup, ref_stream)
        nccl_results[("all_gather", nbytes)] = n
    for total in sizes:
        if total < world: continue
        inp = torch.empty(total, dtype=torch.uint8, device=device)
        out = torch.empty(total, dtype=torch.uint8, device=device)
        with torch.cuda.stream(ref_stream):
            n = _time_gpu(lambda: dist.all_to_all_single(out, inp), n_iter, n_warmup, ref_stream)
        nccl_results[("all2all", total)] = n
    runtime.shutdown()

    for kind in ("all_gather", "all2all"):
        for nbytes in sizes:
            v = vec_results.get((kind, nbytes))
            t = tma_results.get((kind, nbytes))
            n = nccl_results.get((kind, nbytes))
            if v is None or t is None or n is None:
                continue
            rank0_print(f"{kind:<12} {nbytes:>10} {v:>10.1f} {t:>10.1f} {n:>10.1f} {t/n:>9.2f}")

    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
