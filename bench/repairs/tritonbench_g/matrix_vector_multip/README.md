# matrix_vector_multip: checked FP32 kernel

The required entries follow the `add_value` kernel-only scope: readable source
in `Kernels.lean`, Python and memory preconditions in `Contract.lean`, exact
mathematical kernel correctness in `Correct.lean`, and an auxiliary source-to-native
FP32 observation in `Misc/NativeCorrect.lean`. `FPEquiv.lean` compares that
complete, precision-tagged source with the explicit-FMA candidate, including
a complete arbitrary-order grid and its full output and exterior frame.
Allocation, autotuning orchestration and returned
Tensor metadata are outside these required statements.

## Formula, precision and input contract

The independent target is `out[i] = Σ j : Fin M, inp[i,j] * vec[j]`. The Real
interpretation proves that formula for every active row. It makes no equality
claim between rounded FP32 arithmetic and exact real arithmetic.

The repaired Python checks rank-2 `(N,M)` and rank-1 `(M,)` Tensors, matching
reduction lengths, strided storage, FP32 dtype, the same CUDA device, default
contiguity, and resolved negative-view flags. `N,M < 2**63`; each nonempty actual
data pointer must be divisible by four. Non-Tensors are rejected before reading
attributes. Empty pointers are not queried. All guards precede allocation.
`checkInputs_iff` proves that the Lean checks admit exactly `InputContract`.

Input storage may overlap, and nonzero storage offsets and unusual irrelevant
singleton strides are allowed. The wrapper passes the original input objects
with canonical contiguous strides `(M,1,1,1)`. There is no input materialization.
NaNs, infinities, subnormals and signed zeros are admitted; no finite-value
condition is hidden in the contract.

`KernelConfig` retains all 48 combinations: `BLOCK_N ∈ {1,2,4,8}`,
`BLOCK_M ∈ {32,64,128}`, stages in `{3,4}`, and warps in `{4,8}`. The mathematical
and FP theorems are more general in block sizes; positive `BLOCK_M` is their
only block-size premise. `KernelIndex.config_bounds` connects the actual
configuration set to finite-coordinate safety.

`KernelMemory` describes aligned FP32 word bases, complete-word address bounds,
and output/input separation for one invocation. In the public checked theorems,
`KernelBinding.memory` derives the input bases and bounds from the actual tensors,
checked contiguity/alignment, and shared `Torch.FP32.TensorLaws`. It also derives
the nonempty pointer equations in `KernelContract`; these are not new caller
assumptions. Only an already-supplied output buffer with an active span and input
separation, plus an in-grid signed-i32 representable program ID, remains an
ordinary invocation premise. Empty outputs impose no unused span bound.

`KernelBinding.state` maps A, B and C to one physical byte-addressed FP32 read
map. The shared `direct_read_eq_logical` theorem then derives every active matrix
and vector read from the original tensors, including storage offsets and the
checked absence of negative-view flags. `KernelBinding.input_words` discharges
the FP theorem's input-value premise. A/B aliases are coherent by construction
and are proved by `KernelBinding.aliases`; no A/B disjointness or input copying
is introduced. The checked public entries require no abstract `matrixRead`,
`vectorRead`, or `InputWords` premise. These bridges do not allocate a buffer or
prove a host return.

## Original defect and retained repair

The original Python remains unchanged, SHA256
`3a61edaa69bf03a48eb938a71da1fe302235dd6686dede2ee56050712d3f3551`.
It has a concrete active index defect even on contiguous FP32 inputs:
`N=3`, `M=stride_an=2**30`, row `2` gives signed-i32 offset `-2**31`, where the
correct offset is `2**31`. `KernelIndex.original_overflow` and
`KernelIndex.repaired_offset` check both values. This shape remains admitted;
the corresponding large allocation has not been run.

`matrix_vector_multip_repaired.py` widens metadata, program/lane coordinates,
pointer increments and the column counter before arithmetic, and uses an
unsigned while loop. It retains all configurations, masks, FP32 load casts,
initial +0 accumulator, accumulation expression and row reduction. Unresolved
negative views are separately rejected because their raw pointer exposes the
unnegated storage. Pointer alignment is checked against the actual pointer,
including possible byte-offset storage slices.

The DSL displays the mathematical index/control interpretation of that loop.
`KernelIndex` proves the uint64 interpretation: scalar/program/lane ABI,
unsigned row and column masks, initial and final while guards, modular pointer
rebinding, exact active addresses, and output/input separation at byte
precision. Inactive formed pointers may wrap. No bound on inactive pointer
formation is imposed. The source-binding AST check independently verifies that
the Python repair changes only the reviewed integer/control operations.

## Kernel specifications and Tensor applications

| Entry | Result |
| --- | --- |
| `TensorBridge.mv_checked_kernel_correctness` | Derives the actual input State and memory contract from Tensor laws/checks, then proves every source loop iteration, the full logical row dot product, complete typed memory, active outputs and exterior frame. |
| `TensorBridge.mv_checked_fp32_kernel_equiv` | Derives actual logical FP32 input words from the same physical read map, then proves successful execution of the same tagged source and complete typed-memory equality with the independent ordered FMA/add graph. |
| `Correct.mv_kernel_correctness`, `FPEquiv.grid_explicit_fma_equiv` | Independent Real specification and complete-grid actual program pair; typed `ProgramIO` judgments include access safety, success, outputs and exterior frames. The additional single-program `FPEquiv.explicit_fma_equiv` remains an ordinary theorem in `Misc/TileEquiv.lean`; the native observation remains in `Misc/NativeCorrect.lean`. |
| `TensorBridge.real_kernel_contract_sound`, `TensorBridge.fp32_kernel_contract_sound` | Exact Python admission and the unsigned index/access correspondence from `KernelContract`. |
| `KernelSource.source_binding` | Literal equality of the displayed source and the established full-loop source used by the reusable execution proof. |

`Proofs/InputBinding.lean` contains the reused input-only Tensor view, base,
logical-coordinate and pointer facts. The optional historical Tensor adapter
imports these same declarations. `Proofs/KernelBinding.lean` composes them into
the required kernel State, `KernelMemory`, `KernelContract` and `InputWords`
without importing host, allocator or returned-Tensor modules.

`Proofs/KernelSource.lean` reuses the full-loop `WholeSource` execution proof,
not the old one-block slice. `WholeMath` supplies the independent finite-sum
argument. `Proofs/KernelFP.lean` independently defines the masked native
instruction recurrence and ordered reduction on logical bit words, then proves
its connection to actual typed source execution. The reference scatter is only
the memory effect of the supplied mathematical/native row formula; it never
executes the source. Public non-circularity audits check both formulas.

The complete loop includes `M=0`, for which active rows store the reduction of
the initial zero accumulator. `N=0` admits no in-grid program IDs. Row and
column tails, arbitrary numbers of column chunks, and the final backedge are
covered. B loads use only the column mask, even on inactive matrix rows. All
other typed memory cells are preserved, including cells at masked addresses.
The per-program result covers every legal invocation; it does not prove a
host scheduler or allocate the output.

## FP semantics boundary

Inputs, work casts, accumulation, reduction and output are FP32. The native
model records FMA operand orientation and a complete ordered addition tree.
Each physical accumulator leaf occurs exactly once. Graph certificates contain
only instruction structure, never floating values or desired outputs. The retained
`WholeNativeChoices.Plan` also covers each structural last-writer selection;
the theorem applies to each resolved graph separately.

The source expression `acc += a*b` is interpreted by a lossless staging carrier
whose carry node executes a ternary FMA. It is not identified with separately
rounded multiplication followed by addition. The initial zero, all masked-zero
FMAs, every accumulator carry and every ordered reduction edge remain present.
Different reduction trees and block configurations are not asserted to give
equal FP32 bits. The explicit-FMA candidate preserves the original native carry
graph. The retained source-to-native result remains an auxiliary observation;
the main pair compares two actual source programs.

Native FMA/add behavior and compiler binding to the ordered graph are the
execution-model boundary. The proof works for any supplied primitive functions
on 32-bit words, including all special values; it assumes no associativity,
commutativity, contraction, finite intermediates, canonical NaNs or FP-to-Real
identity. No numerical transformation experiment is pending. Hardware primitive
conformance is not recorded as a deferred rewrite assumption. The structural
judgment is not a registered numerical-rewrite judgment for
`#print_fp_assumptions`; its public declarations use standard-base axiom audits.

`Kernels.lean` imports only the DSL and contains the readable source. Required
proof imports exclude the historical host/allocator/Tensor-return helpers.
The earlier broader endpoints are retained under `Proofs/HistoricalCorrect`
and `Proofs/HistoricalFPEquiv` in explicitly historical namespaces. Their
`Numerical.LocalFP32` adapter is an optional primitive-model interface, not a
required numerical TODO.

## Validation

The complete kernel build and targeted eight-module Tensor-bridge rebuild
validate all 18 required kernel modules and 22 preserved compatibility
modules with Lean 4.29.0 (`LEAN_NUM_THREADS=2`): 40 distinct modules,
62 distinct standard-base axiom checks, and zero compiler errors. The required
import closure contains no historical host, allocator, returned-Tensor, or
`Proofs/TensorBinding` module. Both independent reference formulas passed
non-circularity audits. The focused Python
check passed 22 pre-allocation rejection cases and 434 actual interpreter
launches across all 48 source configurations.

Run the existing source/guard/interpreter check with:

```sh
python bench/repairs/tritonbench_g/matrix_vector_multip/check_contract.py
```

This check uses the real Triton CPU interpreter and real CPU tensor storage;
CUDA allocation/context and autotune selection are mocked. It exercises all
48 configurations, empty extents, multiple chunks, both tails, aliases, storage
offsets, singleton strides, special-value admission, alignment/negative-view
rejection and unchanged input bytes. Small exact integer examples validate the
structure, not native floating arithmetic. No GPU execution or native numerical
conformance is claimed.

`Correct.lean` displays the independent `matrixVector` formula and all fields
of its public IO: the FP32 matrix/vector channels, active output rows and the
actual complete source invocation. `Misc/NativeCorrect.lean` displays the same geometry
with word encoding, the existing ordered native backend and its word observer.
The Real specification and ordinary native theorem compose the shared
`ProgramIO.Implements` relation. The optional `FPEquiv.explicit_fma_equiv`,
retained in [`Misc/TileEquiv.lean`](Misc/TileEquiv.lean), compares the
complete original looping source with a direct explicit `tl.fma(a,b,acc)`
candidate, also exposed as `mv_contiguous_fma_kernel` in the selected Python
file. It preserves the input casts, initial +0, loop order, masked tails and
ordered reduction graph; both actual invocations are safe, succeed and have
equal typed output cells and exterior frames.

The single main specification `FPEquiv.grid_explicit_fma_equiv` covers every complete permutation
of the source grid with a full `[N]` output interface. Its execution and safety
proofs include every program, both tails and the zero-program empty case.

The retained original native lowering already represents `acc += a*b` by a
single native FMA carry. `Proofs/ExplicitFMA.lean` proves that the explicit FMA
constructs exactly that carry throughout the complete loop. Thus this source
optimization makes the intended fusion explicit and avoids an intermediate
product representation; it does not assume that separately rounded multiply
and add equal FMA. The pair uses an empty experimental rule set on the full
original all-word domain. Native primitives and the ordered compiler graph
remain explicit model parameters, as in the existing source interpretation.

`Proofs/AccessSafety.lean` follows every actual loop predecessor state and
changing A/B pointer; its proof covers the prefix, all column chunks and the
final masked store. `Proofs/RealResult.lean` and `Proofs/NativeResult.lean`
retain the earlier raw execution/results and connect them to the public IO.
`Misc/TensorBridge.lean` retains the original checked propositions and derives them
from those same foundations. Neither primary entry imports the bridge or the
Tensor binding implementation in `Proofs/KernelBinding.lean`.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.
