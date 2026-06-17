# gfc — Group-Free Symmetric Collectives

A Python + Triton runtime for diffusion / DiT serving that provides subgroup
collectives (`all_gather`, `all2all`, `p2p_put`, `p2p_get`, `barrier`) over
**arbitrary subsets of the global rank set** — without ever calling
`torch.distributed.new_group`.

After a one-time symmetric-memory rendezvous on the world group, every
collective is performed by Triton kernels that load from / store to remote
peer pointers exposed by PyTorch Symmetric Memory. NCCL is used only for the
initial bootstrap handshake; it is **not** on any collective data path.

See [`docs/design.md`](docs/design.md) for the maintained design overview, with
the detailed design specs under `docs/specs/`.

## Quickstart

```python
# rank 0 / rank 1 / ... — launch with torchrun.
import torch
import torch.distributed as dist
from gfc import (
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
    init_distributed_for_runtime,
)

device = init_distributed_for_runtime()        # init NCCL, set CUDA device
config = SymmetricCollectiveConfig(
    max_group_size=4,
    max_collective_bytes=128 * 1024 * 1024,
)
runtime = SymmetricCollectiveRuntime(config, device=device)

# Register a subgroup. Order matters — `(0, 1, 3)` and `(3, 1, 0)` are
# *different* groups with different group_ids.
g = runtime.register_group((0, 1, 3))

# all_gather across the subgroup. Only ranks in the group participate.
if g.local_index >= 0:
    inp = torch.full((1024,), dist.get_rank(), dtype=torch.bfloat16, device=device)
    out = torch.empty(g.size * 1024, dtype=torch.bfloat16, device=device)
    runtime.all_gather(inp, out, g)
    runtime.stream.synchronize()
    # out[i * 1024 : (i + 1) * 1024] now holds rank `g.ranks[i]`'s input.
```

## Collectives provided

| Method | Direction | Notes |
|--------|-----------|-------|
| `runtime.barrier(g)` | n/a | release/acquire-sys barrier on group `g` |
| `runtime.all_gather(inp, out, g)` | remote → local | pull-model |
| `runtime.all2all(inp, out, g)` | remote → local | pull-model, slice `t` of rank `s`'s input is destined for `g.ranks[t]` |
| `runtime.p2p_put(dst_rank, src)` + `p2p_put_recv` | local → remote | push protocol (sender-owned copy to peer comm_buf) |
| `runtime.p2p_get(src_rank, dst)` + `p2p_get_serve` | remote → local | pull protocol |

## Stream model

Single-stream by design. `runtime.stream` is a dedicated CUDA stream owned by
the runtime; all collectives are enqueued onto it under a process-wide
submit lock. External integration via:

```python
ev = torch.cuda.Event(); ev.record(producer_stream)
runtime.wait_for_external(ev)
runtime.all_gather(inp, out, g)
ev_done = runtime.record_event()
consumer_stream.wait_event(ev_done)
```

On each submit the runtime reads the caller's current stream
(`torch.cuda.current_stream()`) and orders its input staging after it, so an
input produced on your stream is safe to read without a manual fence. You stay
responsible for ordering your *consumer* after the output — use the
`record_event` / `wait_event` handshake above, or `runtime.stream.synchronize()`.
v1 is single-stream; concurrent multi-stream submission is a v2 non-goal.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `SYMM_COLL_DEBUG=1` | Enable the full-world group-consistency check (and session-nonce cross-check) |
| `SYMM_COLL_TIMEOUT_MS=N` | Sets `config.timeout_ms` (default 30000). Arms the in-kernel barrier / fused-receiver watchdog: a spin that exceeds the bound `trap`s (surfaces as a CUDA error) instead of hanging. `0` disables it. |
| `SYMM_COLL_DUMP_SIGNALS=1` | Reserved — parsed but not yet wired to a watchdog/signal dump |
| `SYMM_COLL_USE_TMA=1` | Equivalent to `config.use_tma=True` |
| `SYMM_COLL_USE_COPY_ENGINE=1` | Use host-enqueued `cudaMemcpyAsync` copy-engine path |
| `SYMM_COLL_COPY_SMS=N` | Persistent vec-copy CTA count (default: `config.copy_sms`, 24) |
| `SYMM_COLL_PIPELINE_CHUNKS=N` | Override `config.pipeline_chunks` (stream-split chunked staging) |
| `SYMM_COLL_MAX_PIPELINE_CHUNKS=N` | Override `config.max_pipeline_chunks` (static launch-time ceiling) |
| `SYMM_COLL_ENABLE_FUSED=1` | Use the fused single-kernel data path (`config.enable_fused_path`) |
| `SYMM_COLL_FUSED_NUM_CHANNELS=N` | Override `config.fused_num_channels` (also sizes the `step_pad` grid) |
| `SYMM_COLL_FUSED_CHUNK_SIZE=N` | Override `config.fused_chunk_size` (must be a multiple of `alignment`) |
| `SYMM_COLL_AUTOTUNE_CONFIG=path.json` | Load a per-call path/knob policy table (`config.autotune_config_path`); a fused policy needs `fused_num_channels` provisioned to match — see [Autotune configs](#autotune-configs) |

## Testing

```bash
# Picks up GPUs from $GFC_GPUS (default: 1,2,3,4 so GPU 0 is left alone).
./tests/run_all.sh
```

Individual tests live under `tests/`. All multi-process tests are torchrun
scripts; the harness `tests/_harness.py` initialises NCCL from torchrun env.

## Benchmarks

```bash
torchrun --standalone --nproc_per_node=4 \
    benchmarks/bench_symm_collectives.py \
    --collective all2all --bytes 1048576 --iters 1000 --warmup 100 \
    --dtype bf16 --use-tma 0 \
    --validate --compare-nccl-reference

# Copy-engine path for large messages.
torchrun --standalone --nproc_per_node=4 \
    benchmarks/bench_symm_collectives.py \
    --collective all2all --bytes 16777216 --iters 200 --warmup 50 \
    --dtype bf16 --copy-engine 1 \
    --validate --compare-nccl-reference

# CUDA Graph replay is not a supported runtime API in v1. The benchmark has an
# explicit --cuda-graph 1 experiment for launch-overhead studies only.

# Diffusion preset sweep (small_ctrl / mid_act / large_latent).
./benchmarks/run_diffusion_shapes.sh
```

### Autotune configs

`benchmarks/autotune_collectives.py` sweeps `(path, knobs)` policies and emits a
JSON table; load it at runtime with `SYMM_COLL_AUTOTUNE_CONFIG=path.json` (or
`config.autotune_config_path`).

A fused policy in the table names a `fused_num_channels`, and the runtime sizes
its symmetric `step_pad` grid **once** from `config.fused_num_channels` — a table
that asks for more channels than the runtime allocated is rejected. So you must
provision the runtime to match the table, not just point at it. The runtime
validates this at load and fails fast with the exact knob to set if they differ.
The fused kernel also launches one sender CTA and one receiver CTA per channel,
so `2 * fused_num_channels` must fit on the device SM count; otherwise the
runtime rejects the fused path instead of risking a receiver-spin deadlock.

The shipped `benchmarks/autotune_h100_4r.json` was tuned with 32 channels, so its
larger `all2all` buckets select `fused_num_channels: 32`. The default
`config.fused_num_channels` is 24, so load it with the channel count provisioned
to match:

```bash
SYMM_COLL_AUTOTUNE_CONFIG=benchmarks/autotune_h100_4r.json \
SYMM_COLL_FUSED_NUM_CHANNELS=32 \
torchrun --standalone --nproc_per_node=4 ...
```

## Hardware notes

* Tested on 8× NVIDIA H100 80GB on NVLink P2P (SM 9.0).
* Software baseline (matches the spec): `torch==2.10.0+cu130`, `triton==3.6.0`,
  CUDA 13.0. The package itself only depends on `torch>=2.10` and `triton>=3.6`.
* TMA over peer pointers is gated by `tma_probe.probe_tma_supported`. The
  probe passes on the development hardware; on systems where it fails,
  initialising with `use_tma=True` raises `TMAUnsupportedError` (no silent
  fallback) and `use_tma=False` keeps everything working on the vec path.
