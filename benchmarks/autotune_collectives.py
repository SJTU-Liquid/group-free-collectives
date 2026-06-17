"""Sweep collective configurations and emit an autotune JSON config.

Run on the same world-size you intend to deploy on:

.. code-block:: shell

    torchrun --standalone --nproc_per_node=4 \\
        benchmarks/autotune_collectives.py \\
        --output gfc_autotune.json

The script benches every candidate ``(path, knobs)`` policy across a
range of per-peer ``slice_bytes`` for both ``all_gather`` and ``all2all``,
picks the fastest policy per ``(collective, group_size, slice_bytes)``,
and writes a JSON config compatible with ``gfc.autotune.AutotuneTable``.

Candidate paths covered (extend this list as new kernels land):

* ``vec_pull``     — legacy Triton pull kernel (vec). Knob: ``copy_sms``.
* ``fused``        — fused single-kernel pull (CTA-pair). Knobs:
                     ``fused_num_channels``, ``fused_chunk_size``.
* ``tma``          — TMA-variant pull kernel (vec=128B store). No knobs.
* ``copy_engine``  — host-enqueued ``cudaMemcpyAsync`` per peer.
* ``pipelined``    — chunked stream-split pipeline. Knob:
                     ``pipeline_chunks``.

Per-call dispatch is driven through :class:`gfc.autotune.AutotuneTable`
which the script mutates on the runtime in between benches; production
use loads the produced JSON via ``config.autotune_config_path`` (or
``SYMM_COLL_AUTOTUNE_CONFIG=path.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)
from gfc.autotune import AutotuneTable  # noqa: E402
from gfc.tma_probe import probe_tma_supported  # noqa: E402


# --------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    """One ``(path, knobs)`` policy to bench."""

    name: str
    config: dict[str, Any]

    def as_table(self) -> AutotuneTable:
        return AutotuneTable(
            [
                {
                    "collective": "*",
                    "group_size": "*",
                    "min_bytes": 0,
                    "config": self.config,
                }
            ]
        )


def _fused_channels() -> tuple[int, ...]:
    """Channel counts swept for the fused path (env-overridable).

    The runtime config must allocate ``step_pad`` for at least ``max(...)`` of
    these, otherwise those fused candidates index ``step_pad`` past its bound
    and are rejected by ``_autotune_path_guard``. ``main`` sizes the config
    from this so every swept candidate is actually benchable.
    """
    _ch_env = os.environ.get("GFC_FUSED_CHANNELS")
    return tuple(int(x) for x in _ch_env.split(",")) if _ch_env else (16, 24, 32)


def _build_candidates(tma_supported: bool) -> list[Candidate]:
    cs: list[Candidate] = []
    cs.append(Candidate("vec_pull_sms24", {"path": "vec_pull", "copy_sms": 24}))
    cs.append(Candidate("vec_pull_sms48", {"path": "vec_pull", "copy_sms": 48}))
    for ch in _fused_channels():
        for chunk in (128 * 1024, 256 * 1024, 512 * 1024):
            cs.append(
                Candidate(
                    f"fused_ch{ch}_chunk{chunk // 1024}K",
                    {
                        "path": "fused",
                        "fused_num_channels": ch,
                        "fused_chunk_size": chunk,
                    },
                )
            )
    if tma_supported:
        cs.append(Candidate("tma", {"path": "tma"}))
    cs.append(Candidate("copy_engine", {"path": "copy_engine"}))
    for pc in (2, 3, 4):
        cs.append(Candidate(f"pipelined_{pc}", {"path": "pipelined", "pipeline_chunks": pc}))
    return cs


# --------------------------------------------------------------------- benching


_DTYPE = torch.bfloat16
_DTYPE_BYTES = 2


def _make_buffers(
    collective: str, slice_bytes: int, group_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    n_elems = slice_bytes // _DTYPE_BYTES
    if collective == "allgather":
        inp = torch.empty(n_elems, dtype=_DTYPE, device=device).fill_(0)
        out = torch.empty(group_size * n_elems, dtype=_DTYPE, device=device)
    else:  # all2all
        inp = torch.empty(group_size * n_elems, dtype=_DTYPE, device=device).fill_(0)
        out = torch.empty(group_size * n_elems, dtype=_DTYPE, device=device)
    return inp, out


_BATCH_ITERS = 8


def _bench_call(
    fn: Callable[[], None], stream: torch.cuda.Stream, n_warmup: int = 10, n_iter: int = 30
) -> float:
    """Capture ``_BATCH_ITERS`` calls into a CUDA graph and time replays."""
    # Compile + warm
    for _ in range(2):
        fn()
    stream.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(_BATCH_ITERS):
            fn()
    stream.synchronize()

    for _ in range(n_warmup):
        with torch.cuda.stream(stream):
            g.replay()
    stream.synchronize()

    samples: list[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            g.replay()
        end.record(stream)
        end.synchronize()
        samples.append((start.elapsed_time(end) * 1000.0) / _BATCH_ITERS)
    samples.sort()
    return samples[len(samples) // 2]


# --------------------------------------------------------------------- driver


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True,
                   help="path to write the autotune JSON")
    p.add_argument("--sizes", type=str, default=None,
                   help="comma-separated slice_bytes list; defaults to a log sweep")
    p.add_argument("--collectives", type=str, default="allgather,all2all")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--include-tma", type=int, default=1)
    p.add_argument("--include-copy-engine", type=int, default=1)
    p.add_argument("--no-cuda-graph", action="store_true",
                   help="time without CUDA Graph (much slower; not recommended)")
    args = p.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if args.sizes is not None:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:
        # 1 KiB → 256 MiB, half-power steps.
        sizes = [
            1 * 1024,
            16 * 1024,
            256 * 1024,
            1 * 1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            128 * 1024 * 1024,
            256 * 1024 * 1024,
        ]
    collectives = args.collectives.split(",")
    if rank == 0:
        print(f"autotune sweep: world={world} sizes={sizes} collectives={collectives}",
              flush=True)

    # Use a permissive runtime config so all paths are reachable per-call.
    # ``step_pad`` is sized from ``fused_num_channels``; size it for the largest
    # channel count we sweep so every fused candidate is benchable rather than
    # rejected by the runtime's channel guard.
    cfg = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=max(sizes) * world,
        use_tma=False,
        use_copy_engine=False,
        enable_fused_path=False,
        fused_num_channels=max(_fused_channels()),
    )
    runtime = SymmetricCollectiveRuntime(cfg, device=device)
    grp = runtime.register_group(tuple(range(world)))

    tma_supported = False
    if args.include_tma:
        try:
            tma_supported = probe_tma_supported(runtime)
        except Exception as e:
            if rank == 0:
                print(f"TMA probe failed: {type(e).__name__}: {e}", flush=True)
            tma_supported = False

    candidates = _build_candidates(tma_supported=tma_supported)
    if not args.include_copy_engine:
        candidates = [c for c in candidates if c.config.get("path") != "copy_engine"]
    if rank == 0:
        print(f"  candidates ({len(candidates)}):", flush=True)
        for c in candidates:
            print(f"    {c.name}  {c.config}", flush=True)

    # results[collective][slice_bytes][cand_name] = median us
    results: dict[str, dict[int, dict[str, float | None]]] = {
        c: {s: {} for s in sizes} for c in collectives
    }

    for cand in candidates:
        # The runtime was built with use_tma=False, so ``tma_enabled`` is False
        # and ``_autotune_path_guard`` would reject every ``tma`` candidate. The
        # probe above already verified peer-pointer TMA on this hardware, so flip
        # the gate on *only* while the TMA candidate is benched. Toggling it
        # per-candidate (rather than globally) keeps the pipelined candidates
        # honest: ``_effective_pipeline_chunks`` collapses to a single chunk when
        # ``tma_enabled`` is set, which would silently bench them as vec_pull.
        runtime.tma_enabled = tma_supported and cand.config.get("path") == "tma"
        runtime.autotune = cand.as_table()
        for coll in collectives:
            for size in sizes:
                if coll == "all2all" and (size % world) != 0:
                    continue
                err_msg = None
                try:
                    inp, out = _make_buffers(coll, size, world, device)
                except Exception as e:
                    inp = out = None
                    err_msg = f"{type(e).__name__}: {e}"
                errs: list[str | None] = [None] * world
                dist.all_gather_object(errs, err_msg)
                if any(errs):
                    if rank == 0:
                        detail = "; ".join(
                            f"rank {r}: {msg}" for r, msg in enumerate(errs) if msg
                        )
                        print(
                            f"  skip {cand.name}/{coll}/{size}: allocation failed: {detail}",
                            flush=True,
                        )
                    results[coll][size][cand.name] = None
                    continue
                assert inp is not None and out is not None
                if coll == "allgather":
                    fn = lambda: runtime.all_gather(inp, out, grp)
                else:
                    fn = lambda: runtime.all2all(inp, out, grp)
                bench_err = None
                t = None
                try:
                    t = _bench_call(
                        fn, runtime.stream, n_warmup=args.warmup, n_iter=args.iters
                    )
                except Exception as e:
                    runtime.stream.synchronize()
                    bench_err = f"{type(e).__name__}: {e}"
                # Agree across ranks before recording: a candidate that failed
                # on *any* rank must be skipped on *all* ranks, otherwise the
                # ranks fall out of step on epoch/edge_seq for the next
                # candidate and the rest of the sweep deadlocks.
                bench_errs: list[str | None] = [None] * world
                dist.all_gather_object(bench_errs, bench_err)
                if any(bench_errs):
                    if rank == 0:
                        detail = "; ".join(
                            f"rank {r}: {msg}"
                            for r, msg in enumerate(bench_errs)
                            if msg
                        )
                        print(
                            f"  skip {cand.name}/{coll}/{size}: {detail}",
                            flush=True,
                        )
                    results[coll][size][cand.name] = None
                    continue
                results[coll][size][cand.name] = t
                if rank == 0:
                    print(
                        f"  {cand.name:30s} {coll:10s} {size:>10d}B  {t:7.2f} us",
                        flush=True,
                    )
        runtime.stream.synchronize()

    # Pick the best candidate per (collective, size).
    best: dict[str, dict[int, tuple[str, float]]] = {c: {} for c in collectives}
    for coll in collectives:
        for size in sizes:
            row = {k: v for k, v in results[coll][size].items() if v is not None}
            if not row:
                continue
            best_name = min(row, key=lambda k: row[k])
            best[coll][size] = (best_name, row[best_name])

    if rank == 0:
        print("\nBest candidate per (collective, slice_bytes):", flush=True)
        for coll in collectives:
            print(f"  --- {coll} ---", flush=True)
            for size in sizes:
                if size not in best[coll]:
                    continue
                name, t = best[coll][size]
                print(f"    {size:>10d}B  {name:30s}  {t:7.2f} us", flush=True)

        # Emit JSON. Bucket adjacent samples with the same winner into a
        # single rule; the resulting rule range spans
        # ``[run_start, next_run_start - 1]`` so there are no gaps between
        # measured sample points (a request between two sample sizes still
        # gets a policy). The first rule starts at 0 and the last rule has
        # no upper bound, extending the lookup to the entire byte range.
        cand_by_name = {c.name: c for c in candidates}
        rules: list[dict[str, Any]] = []
        for coll in collectives:
            sorted_sizes = sorted(best[coll].keys())
            if not sorted_sizes:
                continue
            # Compact into "runs": consecutive samples with the same winner.
            runs: list[tuple[int, str]] = []
            prev_name: str | None = None
            for size in sorted_sizes:
                name = best[coll][size][0]
                if name != prev_name:
                    runs.append((int(size), name))
                    prev_name = name
            for i, (start, name) in enumerate(runs):
                rule: dict[str, Any] = {
                    "collective": coll,
                    "group_size": world,
                    "min_bytes": 0 if i == 0 else int(start),
                    "config": cand_by_name[name].config,
                }
                if i + 1 < len(runs):
                    rule["max_bytes"] = int(runs[i + 1][0]) - 1
                rules.append(rule)

        # Sort rules by collective then min_bytes desc so the most specific
        # (largest size) match wins first. ``AutotuneTable`` returns the
        # first matching rule.
        rules.sort(key=lambda r: (r["collective"], -int(r["min_bytes"])))

        out_obj = {
            "version": 1,
            "platform": torch.cuda.get_device_name(device),
            "host": platform.node(),
            "world_size": world,
            "dtype": str(_DTYPE),
            "generated_unix_ts": int(time.time()),
            "rules": rules,
        }
        with open(args.output, "w") as fh:
            json.dump(out_obj, fh, indent=2)
        print(f"\nwrote {args.output} ({len(rules)} rules)", flush=True)

    runtime.shutdown()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
