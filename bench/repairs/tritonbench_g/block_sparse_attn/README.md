# Checked block_sparse_attn

Current selected public entries: [Contract.lean](KernelOnly/Contract.lean), [Correct.lean](KernelOnly/Correct.lean), [FPEquiv.lean](KernelOnly/FPEquiv.lean). The [contract review](../../../validation/contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

Reference responsibilities (2026-10-23): The ordered source reference and its main-proof helpers are in `KernelOnly/Proofs/Reference.lean`; `KernelOnly/Reference.lean` is an import-only compatibility entry. These declarations remain necessary to the complete source/native execution connection used by the FP pair proof.

The selected [KernelOnly/Correct.lean](KernelOnly/Correct.lean) and
[KernelOnly/Misc/NativeCorrect.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/block_sparse_attn/KernelOnly/Misc/NativeCorrect.lean) retain shared
`ProgramIO.ImplementsOn` judgments for the complete three-axis source grid.
[KernelOnly/IO.lean](KernelOnly/IO.lean) exposes finite CSR/Q/K/V channels,
actual strided addresses, load masks, output coverage, runtime FP32 scale and
selected native lowering. [KernelOnly/Spec.lean](KernelOnly/Spec.lean) supplies
the independent Real quotient and original ordered native recurrence.
Source access safety, successful execution, every typed output and the exact
regional frame are conclusions of these judgments.

The supporting [KernelOnly/Proofs](KernelOnly/Proofs) modules reuse the
existing complete source execution proofs and prove finite input transport,
actual load/store safety and full grid composition. Tensor coordinate,
input-frame and byte-readback applications remain in
[KernelOnly/Misc/TensorBridge.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/block_sparse_attn/KernelOnly/Misc/TensorBridge.lean); their complete
original statements and the supporting `PhysicalKernel` theorems are preserved.

The repair folds every channel block into QK and gives the third grid axis
one output channel block. It therefore covers dimensions beyond the original
hardcoded two-block path. The checked branch uses contiguous FP32 Q/K/V,
16-by-16-by-16 tiles, four warps, three stages, positive divisible grouped
heads, and a fresh FP32 output of shape `[B,H,length,16*blocks]`. Floating
payloads are passed directly; there are no clones or copybacks.

Only int32 CSR metadata is inspected on the host. Row endpoints must lie in
their allocated column layout; each visited column must be a nonnegative
sequence tile. Reversed or equal endpoints mean an empty row. Duplicates
retain multiplicity and visit order. The unchecked-load flags require actual
Q or K/V row padding. Live calls check signed32 formed offsets, complete
storage extents, four-byte alignment, pointer backing, grid axes and fresh
output separation. Read views may alias each other, including mixed raw-word
aliases. Empty batch or sequence allocates the correctly shaped empty output
and returns before launch-only checks.

`CheckedCSR.data_contract` derives every historical CSR integer-node and
enabled-load obligation from the actual endpoint/column checks and header
bounds. `Physical.query_load` and `key_value_load` show that every enabled
source load reads the current caller address. Physical stores read current
memory; separation proves input stability, while packed output injectivity
proves every result word and the exact outside-store byte frame.

`Contract.lean` is the canonical checked-contract entry, and helper proofs
are under `Proofs/`. The historical checked roots remain `CheckedReal.checked_call` in the task-root
`Correct.lean` and `CheckedFP.checked_call` in the task-root `FPEquiv.lean`; the
current selected entries are the `KernelOnly` modules described above. Both consume checked empty/live
dispatch and retain the exact fresh output descriptor. The mathematical target
is the causal weighted formula over CSR occurrences and the full feature
dimension. Empty/reversed rows are zero. The native target is the complete
initial-word online recurrence; all-masked tiles retain source NaNs. No
native-to-Real accuracy theorem is claimed. The FP endpoint proves all four
bytes of each output word and preserves the observed signed metadata.

Both calls consume `SourceBinding.Binding`: actual input routes, trip counts,
CSR key selection, score graph, value loads, initializer, normalization update
and stores. This is an external selected-source/compiler correspondence for
the pinned Python specialization. It is not inferred from a source hash, a
phantom syntax index, or the existence of the eight saved profiles. It contains
local structural graph maps, with no execution-success or final-output law.
The bound fragments execute before the proof rewrites them to the independent
recurrence. The exact compiler reduction trees, owner routes, contractions and
instruction selection must realize this binding.

Numerical assumptions are confined to scalar rounded operations and fixed
eight-pair MMA instructions. The selected B16 profile uses SM90
`mma.sync.aligned.m16n8k8`, with an explicit seed operand; it does not use
WGMMA. No whole dot or reduction is a numerical oracle. GPU instruction
conformance remains unmeasured. Tensor/raw-memory interpretation, observed
metadata, fresh allocation, the local little-endian word codec, absence of
concurrent mutation and successful CUDA launch are ordinary runtime boundaries.
The shared launcher grid-product issue remains deferred in `bench/TODO.md`.

Validation is reproducible with `check_contract.py` (53 CPU interpreter,
guard and source-pin cases) and `check_lowering.py` (eight offline SM90
compilations for one through four channel blocks and both load-mask settings).
CPU facade/interpreter results and offline compilation are not GPU numerical
validation. The Lean entries include axiom, public-statement and target
non-circularity audits.

The public metadata domain retains signed INT32 endpoints/columns, the
original nonnegative first/column conditions and an explicit finite visit
capacity. Reversed endpoint intervals have no visits; duplicate columns
retain occurrence multiplicity. `visitCapacity` proves that every original
physical memory has a sufficient finite capacity, so this size parameter
does not restrict the prior domain. Native floating channels admit all words.

The original `raw_whole_real` and `raw_whole_native` remain ordinary physical
application theorems. Their `RawABI` proves output storage/separation and CSR
conditions; its physical adapter totalizes unallocated input reads. This old
domain is preserved exactly. The stronger original Tensor `ABI` also supplies
actual input backing: `physical_in_bounds` derives every shared IO window,
and `native_physical_inputs`/`real_physical_inputs` bind the finite snapshots
to the caller's actual memory. `physical_program_trace` identifies each
shared invocation with the existing physical store trace, joining the public
proof to the unchanged complete byte/Tensor results through the same source
execution foundation. No Tensor or allocator internals are added to the
public kernel statements.

## Implementation-pair review (2026-10-21)

The preserved scalar-left candidate still needs a bridge to the selected complete feature-block grid, raw scale register and native instruction/replica lowering. The current native theorem proves one source against its ordered formula.
`KernelOnly/FPEquiv.lean` records the missing target or program connection and publishes no pair specification. The retained source-to-native theorem is available separately in `KernelOnly/Misc/NativeCorrect.lean`. Native results retain their declaration names and full input/output contracts. This is a program-proof gap, not an experimental numerical obligation.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `KernelOnly/Misc`:

- `KernelOnly/Misc/NativeCorrect.lean`
- `KernelOnly/Misc/Reader.lean`
- `KernelOnly/Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
