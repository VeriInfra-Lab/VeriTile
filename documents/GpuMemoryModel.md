# GPU Memory Modeling Scope

This document states what VeriTile means when it talks about "GPU memory".
It is a functional-correctness model for the supported Triton subset, not a
microarchitectural model of CUDA hardware.

The named-cell core and the task-specific physical byte-address bridges are
different layers. The sections below describe the core unless explicitly
marked otherwise. Physical access requirements and their external references
are collected in [Physical byte-address contracts](#physical-byte-address-contracts);
compiler observations for a particular kernel remain beside that kernel.

## Modeled Layers

| GPU concept | VeriTile model | Notes |
| --- | --- | --- |
| Global memory / HBM | `BlockState.mem : RegionName -> Nat -> MemCell` | Target of `tl.load` and `tl.store`. `RegionName` separates named buffers; `Nat` is a cell offset, not a byte address. Proof-facing Real contracts use `readMem` / `writeMem`. |
| Block-local variables | `BlockState.regs : RegFile` | Logical Triton SSA/register values created by assignments. This is not the physical CUDA register file. |
| Addressing metadata | `TensorView`, `Offset.strided`, `InputAt` | Connects logical tensors to global-memory offsets for theorem statements and proof internals. |

The main theorem-facing surface is:

```lean
TensorView.loaded s view tensor
TensorView.observe sf view idx
```

`TensorView` is metadata: it records a region, base offset, and per-axis
strides. It does not store data. Data lives in `BlockState.mem`, while existing
Real-valued theorem statements observe it through `BlockState.readMem`.

## Addressing Model

VeriTile currently supports strided-affine tensor views:

```text
addr = base + sum_i idx_i * stride_i
```

This covers the layouts used by the current FA-1 4D views and the usual
contiguous/strided examples. Lower-level proofs can still use `InputAt` with
an arbitrary offset function when a helper has not been packaged as a
`TensorView`.

Partially modeled:

- First-class pointer values for `RegionName × Nat`.
- Triton-style block pointer values for `tl.make_block_ptr` / `tl.advance` and
  checked block-pointer load/store with zero padding / store skip.
  The optional checker enforces metadata rank equality and static
  `tl.advance` no-underflow cases; theorem-side contracts are
  `BlockPtr.WellFormed`, `BlockPtr.CheckedAxesValid`, and
  `BlockPtr.AdvanceNonnegative`.

Not modeled yet:

- Pointer casts, pointer comparison, or pointer alias analysis beyond
  `RegionName` equality.
- Hardware/TMA block-pointer behavior beyond the sequential lane semantics.
- Full paged-KV proof infrastructure. The storage model now supports the
  core indirect-addressing pattern — typed index loads feeding pointer
  arithmetic — and `IndirectView` packages the read-only view layer. Bounds,
  alias/page-ownership, and paged FA-1 equivalence proofs are still future
  consumer work; see issue #1.
- Rich signed/unsigned integer dtype lattice. The current typed HBM model
  supports `.nat` (`tl.uint8/uint16/uint32/uint64`) and mathematical `.int`
  (`tl.int8/int16/int32/int64`), enough for index/block-table cells, but not
  bit-width-specific integer semantics.

## Sequential Consistency

Within one symbolic program instance, memory is sequentially consistent:

- a `tl.store` updates `BlockState.mem`;
- a later `tl.load` from the same `RegionName` and offset observes that update;
- masked stores leave masked-off lanes unchanged;
- masked loads with `other=None` use `BlockState.undef` for masked-off lanes.

This is the right abstraction for current per-program proofs. Whole-grid
execution is only modeled as a theorem surface: `GridIndex` instantiates
`BlockState.pids`, and `Kernel.ForAllPrograms` / `ForAllProgramsSome` quantify
over every program instance in an ND grid. VeriTile does not yet model a
sequential or concurrent launch executor, overlapping writes, races, full CUDA
atomic memory ordering, or scheduling. Whole-grid memory results are available
only through explicit disjoint-frame or atomic/RMW merge relations.

Layer-2a frame reasoning is modeled as a predicate-level proof contract:
`WriteFootprint := (RegionName × Nat) → Prop` and `BlockState.WriteWithin`
state that a single-program execution changed only the cells inside a supplied
footprint. Layer-2b adds `Kernel.mergeFrames`: an extensional, disjoint
whole-grid merge over explicit per-program `Kernel.ExecFrame`s. This is still
not a concurrent/interleaved executor. Overlapping ordinary writes,
scheduling, barriers, async, and shared memory remain outside the model.
Atomics are covered only by the narrow algorithm-level slices described in
[`ConcurrencySemantics.md`](./ConcurrencySemantics.md): atomic-add sum merge
and selected single-cell RMW linearization with explicit witnesses.

## Not Modeled

The following hardware layers and performance effects are intentionally outside
the current semantic contract:

- Shared memory / SMEM allocation and bank conflicts.
- L1/L2 caches, cache modifiers, eviction policy, volatile memory behavior, and
  coalescing.
- Physical register allocation, register pressure, spilling, occupancy, warps,
  lanes, or scheduling.
- Tensor Core / WGMMA instruction behavior and mixed-precision accumulation.
- Async copy, TMA, barriers, fences, or inter-program synchronization.
- Full CUDA atomic memory ordering and cross-block memory races. VeriTile has
  only limited algorithm-level atomic slices with explicit merge or
  linearization witnesses.

These omissions mean VeriTile proves real-valued functional correctness for a
single symbolic Triton program instance. It does not prove performance
properties or CUDA memory-system fidelity.

## Extension Path

The current model is intentionally small. The landed memory-proof layers are:

- **Memory safety / bounds (#1):** active-lane region-bounds contracts on top
  of the current typed storage layer.
- **Write footprints / frame:** predicate-level `WriteFootprint` and
  `BlockState.WriteWithin` contracts for single-program frame reasoning.
- **Disjoint whole-grid composition:** `Kernel.mergeFrames` merges
  explicit per-program `Kernel.ExecFrame`s when their write footprints are
  pairwise disjoint.

Additional implemented proof helpers include:

- **Structured footprint extraction:** `WriteFootprint.tileImage`,
  `activeTileImage`, and address-image helpers derive predicate footprints from
  common direct, masked, checked block-pointer, and regular store patterns.
- **Unrelated-frame helpers:** convenience lemmas prove that cells or
  whole regions outside a single-program or grid footprint are preserved.

### Layout and flat memory: current implementation

`TensorView` expresses base/stride layouts. `KernelIO` signatures record
input/output windows, masks, and their relationship to program ids. The
specific layouts supported by a kernel are determined by its signature.

The flat-memory bridge is implemented in `Memory/Flatten.lean` and
`Memory/FlattenR.lean`. `FlatAlloc` allocation ranges and disjointness
hypotheses transport region-model proofs into flat pointer memory.
`Kernel.TraceSafe` / `TraceSafeR` provide per-execution safety obligations,
including register-indirect addressing. The `⊨` / `⊨[R]` surfaces in
`Memory/KernelSpec.lean` assemble the bridge conditions, termination, outputs,
and frame. See the [vector-add example](../bench/examples/VectorAdd/Correct.lean).

This shared flat-memory bridge still uses typed, element-sized cells. Separate
task-specific byte-address bridges exist, as described below; they do not make
mixed-dtype overlapping allocations or full hardware alias behavior part of
every core theorem. Each theorem retains its allocation and safety assumptions.

Longer-term extension points remain:

- **Paged KV / indirect addressing (#1):** extend the current `IndirectView`
  smoke/proof surface into paged-attention-specific logical views and
  consumer-side equivalence theorems.
- **Async and concurrency (#1):** `ConcurrencySemantics.md` defines the
  boundary for shared-memory state, barriers, atomics, async/TMA, and explicit
  scheduling or trace models.
- **Floating-point fidelity (#1):** replace or refine the `R` abstraction with
  IEEE / mixed-precision semantics where needed.

## Physical byte-address contracts

This is the shared reference for interpreting physical load/store premises.
It does not assert that the named-cell core proves these premises or that a
particular compiler lowering has been formally verified.

### Instruction access size and alignment

[PTX ISA 8.7, section 6.4.1, Addresses as Operands](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#addresses-as-operands)
defines addresses in bytes and requires natural alignment to the total access
size of the instruction. Misaligned behavior is undefined; it need not fault.
Scalar `.b32` accesses require 4-byte alignment, while `ld.v4.b32` accesses
16 bytes and requires 16-byte alignment. A scalar element's size alone does
not justify the alignment or bounds of a vectorized instruction.

For a physical bridge, active accesses also need valid byte spans, an applicable
address space, allocation permissions and the bounds needed to avoid address
wrap. These obligations are separate from the mathematical tensor formula.
They are not replaced by a predicate asserting Tensor contiguity.

### Masks, transfers and external contracts

[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
and [Triton store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
specify that false mask elements do not perform memory accesses. The corresponding
[PTX predicate guards](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#predicated-execution)
control instruction execution. Inactive load values and their possible use in
later computation still need the source-level proof's treatment.

[PTX ld](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
and [PTX st](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)
specify memory transfers. Concrete packing, permissions, allocation freshness,
memory-frame laws and successful race-free execution remain explicit external
contracts wherever the theorem assumes them. A specification citation is not
a proof of the compiler, allocator or GPU runtime implementation.

### Torch binding and current consumer

[`VeriTile/Torch/Tensor.lean`](../VeriTile/Torch/Tensor.lean) owns shared Tensor
metadata observations; [`FP32.lean`](../VeriTile/Torch/FP32.lean) binds FP32
Tensor views to backing storage and logical values. Their API references sit
next to `TensorLaws` and `logicalValue`. This layer imposes no universal storage
alignment and imports no benchmark code. Kernel admission policy belongs to
the per-task contract; native access requirements belong to the applicable
backend bridge.

The current [add_example revision](../bench/repairs/tritonbench_g/add_example/README.md)
reuses the existing finite-address helpers under
`bench/tritonbench_g/add_example/Proofs/` and supplies its own checked host
composition. These remain scoped helpers, not a generic implementation of all
Triton memory instructions. Its
[load/store audit](../bench/repairs/tritonbench_g/add_example/LOAD_STORE_AUDIT.md)
records the source-to-model mapping, Triton version, targets, compiler evidence,
reproduction command and GPU execution status. Those task-specific observations
do not establish the same lowering for other kernels or backends.
