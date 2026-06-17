# Group-Free Symmetric Collectives — Design Spec

> **Status: original design (2026-05-12), partially superseded — historical record.**
> This is the first full design; it is kept for context, not as the current
> contract. Judge current code against [`docs/design.md`](../../design.md) (the
> maintained protocol overview) plus the deltas below, **not** against this
> document verbatim.
>
> Superseded / changed since this spec:
> - **Barrier signalling.** The `arrive`/`finish`-ack two-lane `signal_buf`
>   (§1, §4) is gone. It is now a per-edge **double-buffered** grid
>   `uint64[2 slots, world_size]`: barrier *N* on edge `(self, peer)` uses
>   `slot = edge_seq[peer] & 1`. There is no finish/ack lane (`gfc/kernels/barrier.py`).
> - **Group descriptor.** `group_ranks_pool` + `pool_row` + the `max_groups=64`
>   cap (§1, §2.1) are gone. Each group owns a per-group device `ranks_dev`
>   (`uint32[size]`), allocated at `register_group` and freed at
>   `unregister_group`; the group count is unbounded. See
>   [`2026-06-13-dynamic-group-alloc-design.md`](./2026-06-13-dynamic-group-alloc-design.md).
> - **Release/acquire primitive.** The data/signal paths use inline PTX
>   `st.release.sys` / `ld.acquire.sys` (`gfc/kernels/_ptx.py`), not
>   `tl.atomic_xchg`/`atomic_cas` (§0.2).
> - **Stream handoff.** The runtime *does* read the caller's current stream
>   (`torch.cuda.current_stream()`) and orders input staging after it
>   (`_ingest_user_tensors`); §5.1's "does not call `current_stream()`" no
>   longer holds.
> - **Added since.** A fused single-kernel data path (with the `step_pad`
>   completion grid) and an autotune policy dispatch
>   (`vec_pull`/`fused`/`tma`/`copy_engine`/`pipelined`) exist in the code; both
>   are absent here.

Date: 2026-05-12
Target package: `gfc` (group-free collectives)
Target hardware: NVIDIA H100 SM 9.0, NVLink P2P (probed: 8× H100 80GB on one node)
Software baseline: `torch==2.10.0+cu130`, `triton==3.6.0`, CUDA 13.0

## 0. Goal and non-goals

### 0.1 Goal

A Python + Triton runtime for diffusion / DiT serving that provides subgroup
collectives (`all_gather`, `all2all`, `p2p_put`, `p2p_get`, plus a bare
`barrier`) over arbitrary subsets of the global rank set, **without** creating a
`torch.distributed` process group per subgroup.

After a one-time symmetric-memory rendezvous on the world group, every collective
is performed by Triton kernels that load from / store to remote peer pointers
exposed by PyTorch Symmetric Memory. NCCL is not used in the data plane.

### 0.2 Non-goals (v1)

- Multicast / NVLS reductions (`multimem.*`); `reduce`, `reduce_scatter`,
  `all_reduce` of any flavor.
- Push-model `all_gather` / `all2all`. v1 is pull-only.
- Multi-stream concurrent collective submission from one process.
- dtype-aware reductions (raw byte copy only; itemsize used to choose vector
  width).
- Inline-PTX `st.release.sys.u64 / ld.acquire.sys.u64` fast path. v1 uses Triton
  `tl.atomic_xchg/atomic_cas` with `sem="release"|"acquire"`, `scope="sys"`.
  Inline-PTX is a marked v2 optimization, not a silent path.
- Cross-node (v1 single-node NVLink P2P).
- Tensor parallelism / pipeline parallelism fusion APIs.
- Dynamic registration of new world members after init.

### 0.3 Hard correctness invariants

1. The runtime never calls `dist.new_group`. Only the world process group exists.
2. The runtime never calls `torch.cuda.synchronize()` in library code (debug and
   tests may).
3. Every collective consumes **two barrier epochs** for its `(group_id)`:
   pre uses epoch `k`, post uses `k+1`, host advances `group_epoch[gid] += 2`.
4. A bare `barrier()` consumes **one** epoch and advances `+= 1`.
5. Signals are addressed by **source global rank**, never by group-local index.
6. `group_id` hashes the **ordered** rank list (rank order is part of the
   collective semantics for `all_gather` concat order and `all2all` slice
   mapping).
7. All submissions for a given collective are enqueued onto the same CUDA
   stream (`runtime.stream`) and protected by a host-side submit lock.
8. No silent fallback paths. TMA enabled only after a successful capability
   probe; failure raises.

## 1. Symmetric memory layout

### 1.1 Bootstrap and rendezvous

```python
dist.init_process_group(backend="nccl")            # for the rendezvous handshake only
world_pg = dist.group.WORLD
sm.enable_symm_mem_for_group(world_pg.group_name)
```

The bootstrap process group is used solely for:
- the symmetric-memory rendezvous of our regions,
- broadcasting `session_nonce`,
- debug-mode consistency checks (group registration, ordering of rank lists),
- tests/benchmarks reference comparisons.

It is not used in any collective data path.

### 1.2 Per-rank symmetric regions

All three regions are allocated once with `sm.empty(...) + sm.rendezvous(t, world_pg)`.
There is no re-rendezvous, no `dist.new_group`, and no `cudaMalloc` on the data
path after initialization.

```
comm_buf      : uint8  [max_collective_bytes]
                  - The data plane. The runtime is single-stream and reuses
                    this one buffer after the post barrier.
                  - The payload is reinterpreted by kernels as
                    [max_group_size, slice_bytes] for all_gather / all2all,
                    or raw bytes for p2p.

signal_buf    : uint64 [num_signal_slots, world_size]
                  - num_signal_slots = 2
                  - row 0 is arrive: owner-written locally at column
                    self_global_rank.
                  - row 1 is finish: remote-written ack at the acking rank's
                    global-rank column after the peer's arrive token has been
                    acquired.
                  - A rank waits for all finish acks before overwriting its
                    next arrive token.

debug_hash    : uint64 [num_signal_slots, world_size]   (allocated always,
debug_epoch   : uint64 [num_signal_slots, world_size]    written only when
                                                             SYMM_COLL_DEBUG=1)
                  - Populated alongside the token via relaxed stores so the
                    host watchdog can dump (group_hash, epoch, observed_token)
                    triples across ranks on timeout.
```

Local (non-symmetric) auxiliary tables, allocated as CUDA tensors so kernels
can dynamically index them:

```
group_ranks_pool : uint32 [max_groups, max_group_size]
                    - Row r is the ordered rank list of the group occupying
                      pool slot r. Kernels load this row with tl.load to
                      enumerate peers.
```

There is no host-side pointer table passed to kernels. Peer pointer arrays are
read directly from the symmetric-memory handles:

```python
self.comm_buf_handle   .buffer_ptrs_dev()    # int (device-resident void**)
self.signal_buf_handle .buffer_ptrs_dev()
self.debug_hash_handle .buffer_ptrs_dev()    # only used in debug
self.debug_epoch_handle.buffer_ptrs_dev()
```

These ints are passed to Triton kernels as `tl.uint64` and cast to the right
pointer type inside the kernel.

### 1.3 Default config and footprint print

`SymmetricCollectiveConfig` defaults (all overridable):

```
max_group_size       = 8
max_groups           = 64
max_collective_bytes = 128 MiB
num_signal_slots     = 2
alignment            = 128
use_tma              = False
enable_debug_checks  = False          # also reads SYMM_COLL_DEBUG
timeout_ms           = 30_000         # watchdog only; GPU spins do not time out
```

Init logs (INFO level, all ranks):

```
gfc: per-rank symmetric memory footprint
  comm_buf  : 128.0 MiB
  signal_buf: 2 lanes × 8 ranks × 8 B =     128 B
  debug_buf : 2 × (2 × 8 × 8 B)       =     256 B   (written only when SYMM_COLL_DEBUG=1)
  total per rank                       ≈ 128.0 MiB
```

## 2. Group descriptor, epoch manager, token computation

### 2.1 GroupDescriptor

```python
@dataclass(frozen=True)
class GroupDescriptor:
    group_id:    int                 # hash64(b"gfc-group-v1", tuple(ranks))   — ordered
    ranks:       tuple[int, ...]     # caller-supplied ordered rank list
    local_index: int                 # ranks.index(self_global_rank); -1 if not member
    size:        int
    pool_row:    int                 # row in group_ranks_pool used by kernels
```

`group_id` is over the **ordered** rank tuple. Two different orderings of the
same rank set are two different groups and get independent `group_id`s, epoch
counters, and pool rows. This is required because:

- `all_gather` output is `concat([rank_i_input for rank_i in group.ranks])` —
  swapping rank order swaps output segments.
- `all2all` slice `i` is sent to `group.ranks[i]` — swapping rank order
  swaps destinations.

If an order-insensitive "same membership" check is later wanted, a separate
`membership_hash = stable_hash64(tuple(sorted(ranks)))` may be stored as debug
metadata only. It is never used in tokens or barrier state.

### 2.2 `register_group`

```python
def register_group(self, ranks: Sequence[int],
                   group_id: Optional[int] = None) -> GroupDescriptor:
    ranks = tuple(int(r) for r in ranks)
    assert 0 < len(ranks) <= cfg.max_group_size
    assert len(set(ranks)) == len(ranks)
    assert all(0 <= r < runtime.world_size for r in ranks)

    gid = group_id if group_id is not None else stable_hash64(b"gfc-group-v1", ranks)

    # Idempotent: same ordered ranks → same pool_row.
    if gid in self._registered:
        return self._registered[gid]

    assert len(self._registered) < cfg.max_groups, "max_groups exceeded"
    pool_row = self._allocate_pool_row(ranks)
    local_index = ranks.index(self.rank) if self.rank in ranks else -1
    desc = GroupDescriptor(group_id=gid, ranks=ranks,
                           local_index=local_index, size=len(ranks),
                           pool_row=pool_row)
    self._registered[gid] = desc

    if cfg.enable_debug_checks:
        # Cross-rank consistency: hash the ordered ranks on every rank and
        # compare via TCPStore. Mismatch raises immediately.
        self._dbg_verify_group_consistency(gid, ranks)

    return desc
```

### 2.3 `_EpochManager`

```python
class _EpochManager:
    """Strictly monotonic per-group epoch counter, host-side."""
    _epochs: dict[int, int]   # group_id -> next epoch (uint64)

    def next_for_barrier(self, gid: int) -> int:
        e = self._epochs.get(gid, 0); self._epochs[gid] = e + 1; return e

    def next_pair_for_collective(self, gid: int) -> tuple[int, int]:
        e = self._epochs.get(gid, 0); self._epochs[gid] = e + 2; return e, e + 1
```

Host increments are done **at submission time**, not at GPU-completion time.
Stream FIFO order is what makes this correct: the kernel that uses epoch `k`
is enqueued onto `runtime.stream` before the kernel that uses epoch `k+1`.

### 2.4 Stable hash and session nonce

In-tree splitmix64; no external dependency, no Python `hash()`.

```python
def _splitmix64(x: int) -> int:
    M = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
    return (x ^ (x >> 31)) & M

def stable_hash64(*chunks) -> int:
    h = 0xCBF29CE484222325                 # FNV offset
    for c in chunks:
        if isinstance(c, (bytes, str)):
            data = c.encode() if isinstance(c, str) else c
            for b in data:
                h = _splitmix64(h ^ b)
        elif isinstance(c, (tuple, list)):
            for v in c: h = _splitmix64(h ^ (int(v) & ((1<<64)-1)))
        else:
            h = _splitmix64(h ^ (int(c) & ((1<<64)-1)))
    return h
```

`session_nonce`:

- Rank 0 generates a non-zero `uint64` from `secrets.randbits(64) | 1`.
- Broadcast to all ranks via the bootstrap `TCPStore` (`store.set("gfc/nonce", str(n))`).
- Runtime asserts `session_nonce != 0` on every rank.
- In debug mode, runtime additionally cross-checks every rank received the
  same value (via `store.get` and a TCPStore-side AND reduction).

Token:

```python
def compute_token(session_nonce, group_id, epoch) -> int:
    return stable_hash64(session_nonce, group_id, epoch)
```

The token is computed on the host (Python) and passed as a `uint64` kernel
argument. Tokens for distinct epochs of the same group are guaranteed distinct
because `epoch` enters the mix and `_splitmix64` is a bijection on `u64`.

## 3. Barrier protocol — release/acquire-sys with token equality

### 3.1 The protocol

For one barrier with parameters `(group, epoch)`, executed on every rank in
the group concurrently:

```
token = compute_token(session_nonce, group.group_id, epoch)

arrive (release, scope=sys):
    local_arrive = local_signal_buf_ptr + (ARRIVE * world_size
                                           + self_global_rank) * 8
    store_release(local_arrive, token)

wait arrive (acquire, scope=sys):
    for each peer p in group.ranks:
        peer_arrive = remote_signal_buf_ptrs_dev[p] + (ARRIVE * world_size
                                                       + p) * 8
        spin until load_acquire(peer_arrive) == token

finish ack (release, scope=sys):
    for each peer p in group.ranks:
        peer_finish = remote_signal_buf_ptrs_dev[p] + (FINISH * world_size
                                                       + self_global_rank) * 8
        store_release(peer_finish, token)

wait finish (acquire, scope=sys):
    for each peer p in group.ranks:
        local_finish = local_signal_buf_ptr + (FINISH * world_size
                                               + p) * 8
        spin until load_acquire(local_finish) == token
```

Arrive cells are owner-written. Finish cells are remote-written acks proving
the acking rank acquired the owner's arrive token.

Both writes and waits live in the same kernel (one Triton program; one warp;
`group.size` participating threads). For `group.size <= 16` this is sufficient
and simple; larger groups are out of scope for v1 (`assert group.size <= 16`).

### 3.2 Why this is correct

- `tl.atomic_xchg(..., sem="release", scope="sys")` guarantees that **all prior
  stores from the writing thread** are visible to any observer that reads the
  same address with acquire-or-stronger semantics at `sys` scope. The release
  happens-before relationship covers stores done by the same kernel and, via
  CUDA stream-order, all prior kernels that wrote to local memory the receiver
  will later access.
- `tl.atomic_cas(ptr, token, token, sem="acquire", scope="sys")` does not modify
  memory when the current value equals `token`. Once it returns `token`, the
  acquire fence makes the publisher's prior stores visible to subsequent loads
  in this thread.
- `scope="sys"` is required because the participating GPUs are distinct devices
  on NVLink P2P. `.gpu` scope only orders within one device.

### 3.3 Why arrive/finish acks

Barrier signal cells are reused across groups and epochs. A token alone is not
enough if a fast rank can overwrite an arrive cell before a slow peer has read
the previous token. The barrier therefore uses an owner-written arrive row and
a remote-written finish row. A rank can overwrite its next arrive token only
after every peer has written a finish ack proving it acquired the previous
arrive token.

### 3.4 Debug fields

When `SYMM_COLL_DEBUG=1`, each release-write is preceded by relaxed stores of
`group.group_id` to `debug_hash[signal_slot, self_global_rank]` and `epoch` to
`debug_epoch[signal_slot, self_global_rank]`. On watchdog timeout, every rank
dumps:

- its own `_EpochManager` state,
- `(signal_slot, src) → (observed_token, observed_group_hash, observed_epoch)`
  for every cell of its grid,
- the local token it was spinning for.

The CPU watchdog reads these via `get_buffer(rank=self.rank, …)` — no extra
GPU sync needed because tokens are only meaningful once GPU has stored them,
which by the time of timeout has already happened.

## 4. Kernel inventory

Five Triton kernels in `gfc/kernels/`. All take peer pointer arrays directly
from `_SymmetricMemory.buffer_ptrs_dev()` (a device-resident `void**`).

| Kernel | Direction | Use |
|--------|-----------|-----|
| `barrier_kernel` | both | owner-written arrive + remote-written finish ack |
| `pull_copy_kernel` | remote → local | byte copy from peer comm_buf to local tensor (used by `p2p_get`) |
| `all_gather_pull_kernel` | remote → local | every rank pulls every peer's comm_buf slice |
| `all2all_pull_kernel` | remote → local | every rank pulls slice `[self_local_index]` from every peer's comm_buf |
| `p2p_put_kernel` | local → remote | the only remote-write kernel; `tl.store` to peer comm_buf |

TMA variants (`*_tma_*`) live in `gfc/kernels/tma_paths.py` and share the call
signature; runtime picks vec vs TMA based on `use_tma` + per-call alignment
gates.

### 4.1 Vectorized raw-byte copy

Vector width selection (compile-time `tl.constexpr` per launch):

```
align = gcd(comm_buf_base_alignment, slice_base_offset_alignment,
            user_tensor_base_alignment, slice_bytes)
VEC   = 16 if align >= 16 else 8 if align >= 8 else 4 if align >= 4 else 1
```

For `slice_bytes` not a multiple of `VEC`, the last program handles a masked
tail (`tl.load`/`tl.store` with mask). The kernel must work for any positive
`nbytes`. Diffusion tensors are usually 16B aligned; p2p metadata may not be —
both must be correct.

### 4.2 `barrier_kernel` sketch

```python
@triton.jit
def barrier_kernel(token: tl.uint64,
                   group_size: tl.int32,
                   self_global_rank: tl.int32,
                   world_size: tl.constexpr,
                   group_ranks_ptr,           # uint32* into group_ranks_pool row
                   signal_ptrs_dev: tl.uint64):   # void** (device-resident)
    pid = tl.program_id(0)                    # one program
    tid = tl.arange(0, MAX_GROUP_LANES)        # MAX_GROUP_LANES = 16 (constexpr)
    mask = tid < group_size

    peer = tl.load(group_ranks_ptr + tid, mask=mask, other=0).to(tl.int32)

    # signal_ptrs_dev is a device-resident array of uint64 (each entry is a
    # pointer to that rank's signal_buf base). Treat it as uint64* and load.
    sig_ptrs_as_u64 = signal_ptrs_dev.to(tl.pointer_type(tl.uint64))
    peer_base       = tl.load(sig_ptrs_as_u64 + peer,             mask=mask, other=0)
    self_base       = tl.load(sig_ptrs_as_u64 + self_global_rank)

    # See gfc/kernels/barrier.py for the arrive/finish ack implementation.

    # (debug-only: relaxed stores of group_hash, epoch into the parallel debug
    #  grids — elided in this sketch)

    # wait: my grid, column = peer
    src_addr = (self_base + (cell + peer) * 8).to(tl.pointer_type(tl.uint64))
    done = tl.zeros([MAX_GROUP_LANES], dtype=tl.int1)
    while not tl.all(done | ~mask):
        observed = tl.atomic_cas(src_addr, token, token,
                                 mask=mask & ~done,
                                 sem="acquire", scope="sys")
        done = done | (observed == token)
```

(`MAX_GROUP_LANES = 16` is a `tl.constexpr` ceiling; runtime asserts
`group.size <= 16`. Sketch elides exact Triton pointer-cast idioms.)

### 4.3 `all_gather_pull_kernel`

After pre-barrier, every rank's `comm_buf` holds its own input
(staging copy was on `runtime.stream` before the pre-barrier kernel, so by
release-acquire-sys this is visible to peers post pre-barrier).

For local rank, for each `i in [0, group.size)`:

```
peer = group.ranks[i]
src  = comm_buf_ptrs_dev[peer]
dst  = output + i * slice_bytes
copy slice_bytes from src to dst   (vectorized, masked tail)
```

Grid: `group.size × ceil(slice_bytes / TILE_BYTES)` programs. `TILE_BYTES`
typical 64 KiB; per-program inner loop sweeps with vector width chosen as
above.

### 4.4 `all2all_pull_kernel`

Sender layout convention: rank `s`'s `comm_buf` is partitioned into
`group.size` equal `slice_bytes` chunks. Slice `t` is destined for the rank
with `group.ranks[t]`.

Receiver pull:

```
self_li = group.local_index
for i in [0, group.size):
    peer = group.ranks[i]
    src  = comm_buf_ptrs_dev[peer]
         + self_li * slice_bytes
    dst  = output + i * slice_bytes
    copy slice_bytes from src to dst
```

Same grid / vectorization scheme as `all_gather_pull_kernel`.

### 4.5 `p2p_put_kernel`

The only kernel that issues remote stores in v1. Implemented internally as a
2-rank ordered collective `group = [self_rank, dst_rank]`:

```
register/lookup group [self, dst]   →  pool_row
stage src into local comm_buf       (if not already symmetric)
pre-barrier on group (epoch k)
remote write: copy from local comm_buf into dst's comm_buf
post-barrier on group (epoch k+1)
```

`p2p_get` is `group = [src, self]` with `pull_copy_kernel` between barriers.
Both go through `_EpochManager.next_pair_for_collective`.

This eliminates a separate mailbox protocol. The 2-rank epoch is independent
of any larger group `[…src…dst…]` because `group_id` is over the ordered pair
and differs from any larger group containing those ranks.

## 5. Public API

```python
@dataclass
class SymmetricCollectiveConfig:
    max_group_size: int = 8                 # asserted <= 16 for barrier kernel
    max_groups: int = 64
    max_collective_bytes: int = 128 * 1024 * 1024
    alignment: int = 128
    use_tma: bool = False
    enable_debug_checks: bool = False
    timeout_ms: int = 30_000

    @property
    def num_signal_slots(self) -> int: return 2


class SymmetricCollectiveRuntime:
    """Single-stream runtime. Owns one CUDA stream per device, accessible as
    `self.stream`. All collective submissions are serialized through
    `self._submit_lock` and enqueued onto `self.stream`."""

    def __init__(self, config: SymmetricCollectiveConfig,
                 device: torch.device,
                 world_group: Optional[dist.ProcessGroup] = None): ...

    # ---- group lifecycle ----
    def register_group(self, ranks: Sequence[int],
                       group_id: Optional[int] = None) -> GroupDescriptor: ...

    # ---- integration with external compute streams ----
    def wait_for_external(self, event: torch.cuda.Event) -> None:
        """Insert a one-way edge: runtime.stream waits on `event`."""
    def record_event(self, event: Optional[torch.cuda.Event] = None) -> torch.cuda.Event:
        """Record a completion event on runtime.stream and return it."""

    # ---- collectives (no stream= argument) ----
    def barrier(self, group: GroupDescriptor) -> None: ...
    def all_gather(self, input_: torch.Tensor, output: torch.Tensor,
                   group: GroupDescriptor) -> None: ...
    def all2all  (self, input_: torch.Tensor, output: torch.Tensor,
                   group: GroupDescriptor, *,
                   slice_bytes: Optional[int] = None) -> None: ...
    def p2p_put  (self, dst_rank: int, src: torch.Tensor, *,
                   nbytes: Optional[int] = None) -> None: ...
    def p2p_get  (self, src_rank: int, dst: torch.Tensor, *,
                   nbytes: Optional[int] = None) -> None: ...
```

### 5.1 Caller contract (single-stream model)

Producer side (anything producing collective input on a different stream):

```python
ev = torch.cuda.Event(); ev.record(producer_stream)
runtime.wait_for_external(ev)
runtime.all2all(in_, out, group)
ev_done = runtime.record_event()
consumer_stream.wait_event(ev_done)
# now consumer_stream can read out
```

Producer side, simpler integration (caller chooses to run on the runtime
stream directly):

```python
with torch.cuda.stream(runtime.stream):
    produce_input(in_)
    runtime.all2all(in_, out, group)
    consume(out)
```

The runtime does **not** call `torch.cuda.current_stream()` implicitly.

### 5.2 Per-call validation

Every collective entry:

- `group.local_index >= 0` (rank in group).
- `input.is_cuda` and `output.is_cuda`, device == runtime device.
- `input.is_contiguous()` (v1 strict; future: row-major strided).
- `input.nbytes <= max_collective_bytes` for all_gather/p2p;
  `slice_bytes * group.size <= max_collective_bytes` for all2all.
- `group.size <= max_group_size`.
- dtype known (we look at `element_size()`).
- If `use_tma`: alignment satisfies TMA tile requirements; otherwise per-call
  raise (no silent fallback).

## 6. Buffer lifecycle (single-stream)

Correctness rests on:

- SymmetricCollectiveRuntime holds a process-wide `_submit_lock` and enqueues
  onto `runtime.stream`.
- CUDA streams are in-order. Once a collective's post barrier has been
  enqueued, any later kernel touching `comm_buf` will run after it on the GPU.
- The arrive/finish barrier protocol prevents signal overwrite across groups
  and epochs.

No cudaEvent fencing is needed in v1. Multi-stream concurrent submission would
invalidate single-buffer reuse and is a v2 non-goal.

## 7. TMA capability probe and path selection

### 7.1 The probe (every rank, independent)

```
For self_rank in [0, world_size):
  peer = (self_rank + 1) % world_size

  # producer side: each rank writes a known pattern into its own comm_buf[0]
  PROBE_BYTES = 4096
  pattern[r]  = uint8 buffer of length PROBE_BYTES where pattern[r][i] =
                ((r + 1) * 131 + i * 17) & 0xFF
  runtime.stream: copy pattern[self_rank] into comm_buf[0][0:PROBE_BYTES]
  world barrier on bootstrap PG

  # consumer side: each rank independently builds a TMA descriptor for its
  # OWN access of peer's base pointer
  peer_ptr = comm_buf.buffer_ptrs()[peer]     # Python list entry on this rank
  desc = triton.tools.experimental_descriptor.create_1d_tma_descriptor(
             peer_ptr, size=PROBE_BYTES, tile=TILE, dtype=torch.uint8)

  Launch a tiny Triton kernel:
      x = tl.load_tensor_descriptor(desc, [0])
      tl.store(local_probe, x)
  Verify local_probe == pattern[peer]

  Local result: True/False
World barrier; AND-reduce results across ranks via TCPStore; single bool.
```

Descriptors are **constructed by each consumer rank** for the peer pointer
**it** will access. They are not built by rank 0 on behalf of anyone else.

### 7.2 Path selection

| `use_tma` | probe result | behavior |
|-----------|--------------|----------|
| `False` (default) | not run | vec path, `INFO gfc: TMA disabled` |
| `True` | pass | TMA path, `INFO gfc: TMA enabled (peer-pointer load verified)` |
| `True` | fail | `raise TMAUnsupportedError("set use_tma=False to use vec fallback")` |

Per-call gating (even when TMA is enabled): alignment / tile / nbytes checks.
If unsatisfiable, the per-call path raises `TMARequirementError`. No silent
fallback ever.

### 7.3 TMA data-plane descriptors

For each peer, a 1D descriptor is built once at runtime init (base pointer is
`comm_buf_ptrs[peer]`; size is `max_collective_bytes`; tile is the configured
TMA tile). Descriptors are cached on this rank's host and passed into the
TMA-path kernels.

## 8. Test matrix

All tests are torchrun scripts. `tests/run_all.sh` wraps them. Multi-GPU; no
mocks.

| # | File | nproc | Verifies |
|---|------|-------|----------|
| 1 | `test_init.py` | 2 | rendezvous, `buffer_ptrs_dev` non-zero, sizes agree across ranks |
| 2 | `test_remote_copy_kernel.py` | 2 | standalone `pull_copy_kernel`: rank 0 stores a pattern; rank 1 pulls; bytes match |
| 3 | `test_barrier_repeated.py` | 2 | group `[0,1]`, 1000 bare barriers; epoch increments by 1 each call; tokens strictly distinct |
| 4 | `test_subgroup_barrier.py` | 4 | groups `[0,2]`, `[1,3]`, `[0,1,3]`; monkey-patch `dist.new_group` to fail if called |
| 5 | `test_all_gather.py` | 2, 4 | fp16/bf16/fp32/uint8; input filled with rank id; output segment `i` equals `group.ranks[i]` |
| 6 | `test_all2all.py` | 2, 4 | group sizes 2 and 4 plus subgroup `[0,1,3]`; sender `s` fills its slice `t` with a deterministic byte pattern over `(s, t, byte_offset)`; verify the output slice `i` on every rank matches the pattern for `(group.ranks[i], group.local_index, byte_offset)` |
| 7 | `test_p2p.py` | 2 | put + get across 100 epochs each; auto-staging from non-symm source tensor |
| 8 | `test_overlap_order.py` | 3 | G1=`[0,1]`, G2=`[1,2]`; rank 1 calls G1 then G2; 100 iters; no deadlock |
| 9 | `test_cross_kernel_publication.py` | 2 | per iter: stage random data on `runtime.stream` → pre-barrier → pull → verify; 2000 iters; specifically exercises release/acquire-sys across distinct kernel launches |
| 10 | `test_buffer_reuse.py` | 2 | 200 collectives; check no stale bytes carry across comm-buffer reuse |
| 11 | `test_perf_smoke.py` | 2, 4 | small/medium/large; compare against `dist.all_to_all_single`, `dist.all_gather_into_tensor`; reference only |
| 12 (debug-only) | `test_order_mismatch_watchdog.py` | 3 | intentionally inverted G1/G2 on one rank; expect timeout-then-raise with diagnostic dump; **default skipped** so CI never hangs |

Reset between tests: `runtime.shutdown()` plus a world barrier and an explicit
zeroing of `signal_buf` via `memset32`.

## 9. Benchmarks

`benchmarks/bench_symm_collectives.py`. Example:

```
torchrun --standalone --nproc_per_node=4 \
    benchmarks/bench_symm_collectives.py \
    --collective all2all \
    --bytes 1048576 \
    --group 0,1,2,3 \
    --iters 1000 --warmup 100 \
    --dtype bf16 \
    --use-tma 0 \
    --validate --compare-nccl-reference
```

Flags: `--collective {all2all,allgather,p2p_put,p2p_get,barrier}`, `--bytes`,
`--group-ranks` (default = all ranks), `--iters`, `--warmup`,
`--dtype {bf16,fp16,fp32,uint8}`, `--use-tma {0,1}`, `--validate`, `--debug`,
`--compare-nccl-reference`, `--diffusion-preset {small_ctrl,mid_act,large_latent}`.

Output (per rank, plus a rank-0 aggregate line):

```
config: world=4 group_size=4 collective=all2all bytes=1048576 dtype=bf16
        use_tma=0 path=vec
iters: 1100 (warmup 100)
latency           p50=…us p90=…us p99=…us
kernel_time       p50=…us
effective_bw      …GB/s   ((group_size-1) * bytes per iter for all2all)
nccl_reference    p50=…us
group_id          0x{ordered_group_id_hex}
```

`benchmarks/run_diffusion_shapes.sh` runs a preset sweep at sizes
representative of diffusion serving (small control tensor, mid activation
shard, large latent/head shard).

## 10. Repository layout

```
GroupFree-Collective/
├── gfc/
│   ├── __init__.py                  # public exports
│   ├── config.py                    # SymmetricCollectiveConfig
│   ├── group.py                     # GroupDescriptor, stable_hash64, _EpochManager
│   ├── runtime.py                   # SymmetricCollectiveRuntime
│   ├── buffer.py                    # symm-mem region allocation
│   ├── env.py                       # SYMM_COLL_* env handling
│   ├── logging.py                   # INFO/DEBUG, footprint print
│   ├── tma_probe.py
│   └── kernels/
│       ├── __init__.py
│       ├── _common.py               # VEC selection, alignment, ptr packing
│       ├── barrier.py
│       ├── copy_pull.py
│       ├── all_gather.py
│       ├── all2all.py
│       ├── p2p_put.py
│       └── tma_paths.py
├── tests/                           # see Section 8
├── benchmarks/
│   ├── bench_symm_collectives.py
│   └── run_diffusion_shapes.sh
├── docs/
│   ├── design.md                    # short mirror, links to spec
│   └── specs/2026-05-12-symmetric-collectives-design.md
├── pyproject.toml                   # package gfc
├── README.md
└── .gitignore
```

## 11. Environment variables

| Variable | Meaning |
|----------|---------|
| `SYMM_COLL_DEBUG=1` | Enable debug grids, group consistency checks, watchdog dumps |
| `SYMM_COLL_TIMEOUT_MS=N` | Watchdog timeout (default: `config.timeout_ms`) |
| `SYMM_COLL_DUMP_SIGNALS=1` | On watchdog fire, dump full signal grids to stderr |
| `SYMM_COLL_USE_TMA=1` | Equivalent to `config.use_tma=True` |

## 12. Risks and open questions

- **TMA over peer pointers** is unverified. `tma_probe.py` is the gate; if the
  probe fails on H100 + NVLink P2P, we ship vec-only and note this in
  `docs/design.md`. No silent fallback.
- **`tl.atomic_cas(..., sem="acquire", scope="sys")` codegen**: we will inspect
  the generated SASS once on H100 to confirm an acquire load is emitted (or an
  acquire-CAS that suffices). If the path is suboptimal, the spin still
  satisfies correctness; the v2 inline-PTX path will be a perf optimization.
- **`_SymmetricMemory.barrier`** (the built-in world-scope NCCL-style barrier)
  is used only at init / teardown / between tests, never in collective data
  paths.
- **Group registration race**: `register_group` is idempotent per ordered
  rank tuple. Two ranks must register identical ordered ranks for the same
  collective; debug mode enforces this via a TCPStore consistency check at
  registration time.
- **Cross-node**: out of scope. The current `_SymmetricMemory` backend on
  NVLink-only deployments may not transparently extend to RDMA fabrics.
- **`max_collective_bytes`** dominates comm-buffer memory; the init log
  surfaces this.

## 13. Implementation order (informational; the plan goes in the writing-plans phase)

1. Skeleton: package layout, config, group, stable hash, _EpochManager (unit
   tests for hashing/epoch on one process).
2. Bootstrap + symm-mem allocation + footprint print (`test_init.py`).
3. `pull_copy_kernel` and validation harness
   (`test_remote_copy_kernel.py`).
4. `barrier_kernel` and `test_barrier_repeated.py` /
   `test_subgroup_barrier.py`.
5. `all_gather_pull_kernel` + tests.
6. `all2all_pull_kernel` + tests.
7. `p2p_put_kernel` / `p2p_get` via the 2-rank collective path + tests.
8. `test_cross_kernel_publication.py`, `test_overlap_order.py`,
   `test_buffer_reuse.py`, `test_perf_smoke.py`.
9. Benchmark script + diffusion preset sweep.
10. `tma_probe.py` + TMA variant kernels (gated experimental).
11. Documentation pass (`docs/design.md`, README quickstart, env-var section).

## 14. Glossary

- **Symmetric memory**: A memory region for which every participating GPU has
  a remote pointer to every other participant's local allocation, exposed via
  `_SymmetricMemory.buffer_ptrs_dev()`.
- **Signal lane**: One row of `signal_buf`. Row 0 is arrive; row 1 is
  finish/ack.
- **Token**: A 64-bit value `hash64(session_nonce, group_id, epoch)` written
  atomically with release semantics and waited on with acquire semantics.
- **Epoch**: Per-`group_id` monotonic counter advancing by 1 per barrier and
  by 2 per collective.
- **Group**: An ordered tuple of global ranks. The order is part of the
  collective semantics and is baked into `group_id`.
