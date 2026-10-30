# matmul_triton2: masked autotuned FP32/TF32 multiplication

The original benchmark is unchanged. This revision repairs missing M/N input
masks and finite-width indexing, with an explicit contiguous FP32 contract.
The mathematical and source-precision proofs cover allocation, all successful
autotune trials, the final selected launch and the actual returned modeled Tensor.

## Supported contract and repair

`matmul_triton2_repaired.py` accepts two strided, rank-two FP32 CUDA tensors
on the same device, with compatible matrix dimensions. Both are contiguous
and have resolved negative view bits. Each of M, N and K is less than `2**63`.
Each nonempty input has a data pointer divisible by four; empty input pointers
are neither queried nor constrained by alignment.

Original tensors are passed directly with canonical dense strides
`K,1,N,1,N,1`. Contiguous offset views, overlapping inputs, irrelevant unusual
singleton strides, empty matrices, K=0 and all FP32 payloads are included.
The original explicit FP32 output allocation and all eight autotune configs,
warps, stages and keys are retained. Empty M or N returns the allocation without
launching; other launches run in the input device context. No input copies or
general-stride compatibility layer are added.

For M=16, N=256 and K=64, every original configuration has a tile taller than
M. The original K-only mask permits A row 16 at element offset 1024, exactly
past the allocation's 1024 elements. The repair adds M/N masks to the two loads
and retains the existing output mask. The original grouped mapping is valid:
`pid % group_size_m` gives a cyclic row permutation even in the last smaller
group. It is preserved.

Runtime dimensions/strides, program ID and lanes are widened to uint64 before
arithmetic. The result of grouped `min` is explicitly cast too, because its
constexpr branch can otherwise introduce an int32 operand into unsigned
division/remainder. Default TF32 dot precision and the live accumulator are
retained.

## Proof entries

All names below are in `VeriTile.Bench.Repairs.MatmulTriton2.Revision001`.

| Entry | Strong public result |
| --- | --- |
| `Correct.lean` | `matmul_triton2_autotuned_real`: independent full-K mathematical sum of the original Tensor logical values. |
| `FPEquiv.lean` | `matmul_triton2_autotuned_native`: selected configuration's exact padded source graph with local instruction ports. |
| `FPEquiv.lean` | `matmul_triton2_autotuned_tf32_if_local`: transport to independent local TF32 instruction references under explicit correspondence. |

The per-configuration results `matmul_triton2_checked_real` and
`matmul_triton2_checked_native` are also public. Earlier logical/grid/host
results are preserved with their stated scope.

`Contract.lean` specializes matmul1's shared exact admission contract; the
shared `checkInputs_iff` and `checkTensorInputs_iff` prove equivalence with the
named supported domain. `Kernels.lean` contains only source definitions. Real
and FP public entries independently consume helpers without importing one
another.

The shared `FiniteIndex`/`FiniteExecution` helpers prove and execute uint64
casts, cdiv, grouped coordinates, padded lanes, loop comparisons and increments,
remaining-K masks, pointer recurrence and active four-byte address windows.
`FiniteSource` binds this execution to matmul2's complete typed source and
reuses the complete grid coverage and unique-store proof. `TensorBinding`
uses checked alignment, original Torch storage offsets and actual data pointers
to establish readback from original logical values, including aliased inputs.
`InputAdapter`, `FiniteHost`, `CheckedHost` and `LogicalInputs` compose exact
admission, finite execution, allocation and independent references.

`Autotune.Plan` contains a list of trial configurations and a final selection.
Trials may repeat; an empty list represents a cache hit. The proof executes all
trials against the same fresh output, preserves input values and the caller
frame through each trial, then proves the final selected result. Arithmetic
equality between different configurations is never assumed.

The postcondition includes output rank/shape, FP32 dtype, same CUDA device,
contiguity, resolved negative flag, fresh object identity, every output value
and the exterior frame. Allocation is pointwise successful allocation for the
actual call, including storage capacity. Successful-launch premises provide
representable grids (`grid < 2**31`) for executed configurations; this is not
an extra Python input-size cap. Ordinary Tensor/storage laws, fresh allocation
and successful race-free runtime execution remain explicit boundaries. The
shared large-grid launcher compatibility item stays deferred under
`bench/TODO.md`.

## Precision and validation boundary

The precision graph records the selected Triton 3.7.1 SM90 lowering. Configs
0–5 use warp-group TF32 MMA; configs 6–7 use synchronous TF32 MMA. Each local
`MMA8` port consumes exactly eight raw FP32 inputs from each operand, in TF32
input mode, plus a live FP32 seed, and produces an FP32 result. Ascending
eight-term chunks, all padded lanes and the selected configuration are retained.
No unverified binary rounding tree replaces an instruction's internal behavior.
This hardware-family selection is specific to the captured target.

`Native.Correspondence` states three local laws: FP32 zero, one sync MMA8 and
one warp-group MMA8 against separately supplied reference ports. The domain
is all raw FP32 words and FP32 seeds, including NaNs, infinities, signed zeros
and subnormals. Exact TF32 conversion, accumulation, exceptional-value and FTZ
behavior remain obligations for the later numerical experiment batch. The
conditional theorem derives the complete wrapper result; no whole-dot, tile
or output equality is assumed. The structural native result needs no numerical
correspondence. The Real theorem is a separate real-arithmetic interpretation.

Targeted Lean 4.29.0 builds and `#axiomsClean` checks cover the current proof
closure and retained reader entry, with no `sorry` or new axiom declaration.
Run the shared focused test from the repository root:

```sh
python bench/repairs/tritonbench_g/matmul_triton1/revision001/check_contract.py 2
```

It passed 24 pre-allocation rejection cases and 107 CPU-interpreted cases
covering all eight configs. It checks source ASTs, unchanged autotune settings,
guards, masks, casts, original failure witnesses, tails and final groups,
K=0, empty/offset/overlapping inputs, singleton strides, NaN/Inf admission,
input preservation and exact returned allocation. CUDA launch/allocation are
mocked; interpreter arithmetic is not GPU conformance evidence.

Offline SM90 compilation passed all eight configs with i32 and i64 runtime
metadata, confirming the expected TF32 instruction families. Together with
matmul1, 18 cases passed. Torch was 2.13.0+cu130 and Triton 3.7.1. No GPU
execution was possible, and no cross-architecture or native numerical
conformance claim is made.
