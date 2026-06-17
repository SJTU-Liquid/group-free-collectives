"""Benchmark repeated collectives over many different GFC groups.

Rank 0 generates the same style of random rank-mask schedule used by
``tests/test_barrier_stress.py``. Every rank receives the schedule, registers
each unique group, and then repeatedly walks the schedule. Ranks that are not
members of a scheduled mask skip that collective.

The reported latency is based on the max wall time across ranks for one full
schedule walk, divided by the number of scheduled collectives. Throughput is
the aggregate remote bytes moved by all member ranks over that same max time.

Run:
    torchrun --standalone --nproc_per_node=4 \\
        benchmarks/bench_multigroup_collectives.py \\
        --collective allgather --bytes 1048576 \\
        --schedule-len 256 --iters 20 --warmup 3 --copy-engine 1

For allgather, ``--bytes`` is the per-rank input size. For all2all, ``--bytes``
is the per-peer slice size, so each member rank sends ``group_size * bytes``.
"""

from __future__ import annotations

import argparse
import os
import random
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
from gfc.tma_probe import probe_tma_supported  # noqa: E402


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "uint8": torch.uint8,
}


def _mask_to_ranks(mask: int, world: int) -> tuple[int, ...]:
    return tuple(r for r in range(world) if mask & (1 << r))


def _broadcast_schedule(
    *,
    world: int,
    n: int,
    seed: int,
    min_group_size: int,
    max_group_size: int,
) -> list[int]:
    if dist.get_rank() == 0:
        rng = random.Random(seed)
        candidates = [
            mask
            for mask in range(1, 1 << world)
            if min_group_size <= mask.bit_count() <= max_group_size
        ]
        if not candidates:
            raise ValueError(
                f"no valid masks for world={world}, "
                f"min_group_size={min_group_size}, max_group_size={max_group_size}"
            )
        masks = [rng.choice(candidates) for _ in range(n)]
    else:
        masks = [0] * n

    obj_list: list = [masks]
    dist.broadcast_object_list(obj_list, src=0)
    out = list(obj_list[0])
    assert len(out) == n
    assert all(1 <= m < (1 << world) for m in out), "bad mask in broadcast schedule"
    return out


def _percentiles(samples_us: list[float]) -> tuple[float, float, float]:
    s = sorted(samples_us)
    n = len(s)
    return s[n // 2], s[min(n - 1, int(n * 0.9))], s[min(n - 1, int(n * 0.99))]


def _member_call_count(rank: int, schedule: list[int]) -> int:
    self_bit = 1 << rank
    return sum(1 for mask in schedule if mask & self_bit)


def _remote_bytes_per_schedule(schedule: list[int], payload_bytes: int) -> int:
    total = 0
    for mask in schedule:
        group_size = mask.bit_count()
        total += group_size * (group_size - 1) * payload_bytes
    return total


def _a2a_value(sender: int, dst_local_idx: int) -> int:
    return (sender * 17 + dst_local_idx) % 251


def _validate_allgather(group, out: torch.Tensor, n_elems: int) -> None:
    for seg_idx, peer in enumerate(group.ranks):
        seg = out[seg_idx * n_elems : (seg_idx + 1) * n_elems]
        if not torch.all(seg == peer):
            raise AssertionError(
                f"allgather mismatch group={group.ranks} seg={seg_idx} peer={peer}"
            )


def _validate_all2all(group, out: torch.Tensor, n_elems: int) -> None:
    for seg_idx, peer in enumerate(group.ranks):
        expected = _a2a_value(peer, group.local_index)
        seg = out[seg_idx * n_elems : (seg_idx + 1) * n_elems]
        if not torch.all(seg == expected):
            raise AssertionError(
                f"all2all mismatch group={group.ranks} seg={seg_idx} peer={peer}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collective", choices=["allgather", "all2all"], default="allgather")
    p.add_argument("--bytes", type=int, default=1024 * 1024)
    p.add_argument("--schedule-len", type=int, default=256)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--max-group-size", type=int, default=None)
    p.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    p.add_argument("--use-tma", type=int, choices=[0, 1], default=0)
    p.add_argument("--copy-engine", type=int, choices=[0, 1], default=0)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        os.environ["SYMM_COLL_DEBUG"] = "1"
    if args.use_tma and args.copy_engine:
        raise ValueError("--use-tma and --copy-engine are mutually exclusive")
    if args.schedule_len <= 0:
        raise ValueError("--schedule-len must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if not (2 <= world <= 16):
        raise ValueError(f"benchmark expects 2 <= nproc <= 16, got {world}")

    max_group_size = args.max_group_size if args.max_group_size is not None else world
    if not (1 <= args.min_group_size <= max_group_size <= world):
        raise ValueError(
            f"invalid group-size bounds: min={args.min_group_size}, "
            f"max={max_group_size}, world={world}"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dtype = _DTYPES[args.dtype]
    elt = torch.empty((), dtype=dtype).element_size()
    if args.bytes % elt != 0:
        raise ValueError(f"--bytes={args.bytes} must be divisible by dtype size {elt}")

    schedule = _broadcast_schedule(
        world=world,
        n=args.schedule_len,
        seed=args.seed,
        min_group_size=args.min_group_size,
        max_group_size=max_group_size,
    )
    unique_masks = sorted(set(schedule))
    largest_group = max(mask.bit_count() for mask in unique_masks)
    max_collective_bytes = args.bytes
    if args.collective == "all2all":
        max_collective_bytes *= largest_group
    max_collective_bytes = max(max_collective_bytes, 4 * 1024 * 1024)

    config = SymmetricCollectiveConfig(
        max_group_size=largest_group,
        max_collective_bytes=max_collective_bytes,
        use_tma=bool(args.use_tma),
        use_copy_engine=bool(args.copy_engine),
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    if args.use_tma:
        ok = probe_tma_supported(runtime)
        if not ok:
            raise RuntimeError("TMA probe failed; rerun with --use-tma 0")

    mask_to_group = {}
    for mask in unique_masks:
        mask_to_group[mask] = runtime.register_group(_mask_to_ranks(mask, world))

    n_elems = args.bytes // elt
    buffers_by_size: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for group_size in sorted({mask.bit_count() for mask in unique_masks}):
        if args.collective == "allgather":
            inp = torch.full((n_elems,), rank, dtype=dtype, device=device)
            out = torch.empty(group_size * n_elems, dtype=dtype, device=device)
        else:
            inp = torch.empty(group_size * n_elems, dtype=dtype, device=device)
            for dst_local_idx in range(group_size):
                value = _a2a_value(rank, dst_local_idx)
                inp[dst_local_idx * n_elems : (dst_local_idx + 1) * n_elems].fill_(value)
            out = torch.empty_like(inp)
        buffers_by_size[group_size] = (inp, out)

    def run_one(group) -> None:
        inp, out = buffers_by_size[group.size]
        if args.collective == "allgather":
            runtime.all_gather(inp, out, group)
        else:
            runtime.all2all(inp, out, group, slice_bytes=args.bytes)

    def walk_schedule(post_op: Callable[[object], None] | None = None) -> None:
        self_bit = 1 << rank
        for mask in schedule:
            if not (mask & self_bit):
                continue
            group = mask_to_group[mask]
            run_one(group)
            if post_op is not None:
                runtime.stream.synchronize()
                post_op(group)

    if args.validate:
        def validate_group(group) -> None:
            _, out = buffers_by_size[group.size]
            if args.collective == "allgather":
                _validate_allgather(group, out, n_elems)
            else:
                _validate_all2all(group, out, n_elems)

        walk_schedule(validate_group)
        dist.barrier(group=runtime.world_pg)

    for _ in range(args.warmup):
        walk_schedule()
        runtime.stream.synchronize()
    dist.barrier(group=runtime.world_pg)

    schedule_samples_us: list[float] = []
    for _ in range(args.iters):
        dist.barrier(group=runtime.world_pg)
        t0 = time.perf_counter()
        walk_schedule()
        runtime.stream.synchronize()
        elapsed_us = (time.perf_counter() - t0) * 1e6

        elapsed_tensor = torch.tensor([elapsed_us], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=runtime.world_pg)
        schedule_samples_us.append(float(elapsed_tensor.item()))

    sched_p50, sched_p90, sched_p99 = _percentiles(schedule_samples_us)
    per_collective_p50 = sched_p50 / len(schedule)
    per_collective_p90 = sched_p90 / len(schedule)
    per_collective_p99 = sched_p99 / len(schedule)

    remote_bytes = _remote_bytes_per_schedule(schedule, args.bytes)
    throughput_gbps = (remote_bytes / (sched_p50 * 1e-6)) / 1e9
    avg_group_size = sum(mask.bit_count() for mask in schedule) / len(schedule)
    member_calls = _member_call_count(rank, schedule)

    if rank == 0:
        path = "copy_engine" if args.copy_engine else ("tma" if args.use_tma else "vec")
        print(
            f"config: world={world} collective={args.collective} bytes={args.bytes} "
            f"dtype={args.dtype} schedule_len={len(schedule)} iters={args.iters} "
            f"warmup={args.warmup} seed={args.seed:#x} "
            f"group_size_range=[{args.min_group_size},{max_group_size}] "
            f"unique_groups={len(unique_masks)} avg_group_size={avg_group_size:.2f} "
            f"path={path} copy_sms={runtime.copy_sms}"
        )
        print(
            f"rank0_member_calls_per_schedule={member_calls} "
            f"aggregate_remote_bytes_per_schedule={remote_bytes}"
        )
        print(
            "latency_per_collective "
            f"p50={per_collective_p50:.1f}us "
            f"p90={per_collective_p90:.1f}us "
            f"p99={per_collective_p99:.1f}us"
        )
        print(
            "latency_per_schedule   "
            f"p50={sched_p50:.1f}us "
            f"p90={sched_p90:.1f}us "
            f"p99={sched_p99:.1f}us"
        )
        print(
            f"aggregate_throughput {throughput_gbps:.2f} GB/s "
            f"({remote_bytes} remote bytes/schedule at p50)"
        )

    runtime.shutdown()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
