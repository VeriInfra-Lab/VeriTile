# Autotuned matmul: kernel-only specification

The selected source is `matmul_triton_autotune_view_safe.py`, SHA-256
`57ef247c2cd52e4300a270ac83707881b15e11d067dad81430c5d0bde1f75e08`.
This revision changes proof organization and endpoints; the Python source and
its checked domain are unchanged. The original benchmark remains separate.

`Kernels.lean` holds the one readable Triton quotation, shared by both proofs.
`Contract.lean` describes checked admission and ordinary Tensor/launch facts.
`Correct.lean` proves `Kernel.mathematical_kernel`; `Misc/NativeCorrect.lean` retains
`Kernel.floating_kernel` as an ordinary source-to-native theorem. `FPEquiv.lean`
proves `Kernel.matmul_equiv` for the original and existing safe-address source. `IO.lean` displays both strided FP32 input channels,
the FP16 output window, the exact selected source and complete launch schedule.
The public endpoints use shared `ProgramIO` judgments over logical inputs;
actual source trace safety, successful execution, every typed output and the
exact exterior cell frame are their conclusions. Helpers live in `Proofs/`.
`Misc/TensorBridge.lean` preserves the complete Tensor applications
and physical byte/address results as ordinary theorems.

The independent Real target is
`z[i,j] = sum_k A[i,k] * B[k,j]`, followed by `z` when activation is not exactly
`"leaky_relu"`, otherwise `z` for `z >= 0` and `(1/100)*z` for negative `z`.
Real casts are identity. There is no arbitrary `roundHalf` argument in the
new public theorem and no assertion that native arithmetic equals this Real sum.

The checked domain remains rank-two strided FP32 inputs on the same CUDA/HIP
device, matching K, contiguous A, resolved negative/conjugate views,
nonnegative strides, four-byte-aligned nonempty input pointers, and a builtin
activation string. B may be transposed, sliced or broadcast. Input aliases,
storage offsets, inert singleton strides, empty M/N and K=0 remain admitted.
The wrapper forwards the original input objects without copies. Python
classification of Tensor/rank/signed strides/string values is represented by
`Arguments`; the admitted branch has actual Tensor objects and Nat strides.
`checkArguments_iff` connects its Boolean check to the public input contract.

All original configurations are retained:

| Configuration | BM | BN | BK | Group M | Stages | Warps | Waves/EU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA 128 | 128 | 256 | 64 | 8 | 3 | 8 | — |
| CUDA 64 | 64 | 256 | 32 | 8 | 4 | 4 | — |
| HIP 128 | 128 | 256 | 16 | 1 | 2 | 4 | 2 |

`Proofs/KernelTensor.lean` binds source reads to `Tensor.physicalIndex`, actual
storage offsets and data pointers. Both inputs use one physical reader, so
input alias consistency is derived. Every active output store is connected to
the actual contiguous FP16 Tensor address. Output bytes are reconstructed from
the executed source state; there is no caller-supplied final-heap or output
predicate. The conclusions include output values, preservation of both complete
input backing allocations, the exterior byte frame, logical-cell frame,
complete-grid execution and signed integer/formed-pointer safety.

Successful allocation and signed metadata bounds are ordinary Tensor facts:
input/output backing allocations occupy fewer than `2^63` bytes, signed strides
fit, output is fresh and contiguous, and program IDs fit the launcher ABI.
These are separate from Python guards, not hidden padded-address assumptions.
Runtime grid-product compatibility remains the project's separately deferred item.

The floating theorem admits every FP32 bit pattern. It retains all padded
lanes, source block order and incoming FP32 carry, optional ordered comparison,
left-side FP32 multiplication by `0x3c23d70a`, and mandatory FP16 conversion.
The former six `PrimitiveLaws` are now proved from selected definitions.
Only fixed K8 CUDA MMA and K4 HIP MFMA instruction semantics remain parameters;
there is no variable-size dot or whole-result callback. See `Model.md` for
sources and the precise compiler/hardware boundary. No new algorithmic
transformation or numerical experiment assumption is introduced.

Validation for this revision: targeted Lean builds and standard-base axiom
checks pass for both public endpoints and their Tensor/address helpers. The
existing CPU check passes 69 actual-source interpreter calls and nine rejection
classes across all three configurations, plus launch-device restoration checks.
Nine offline Triton 3.7.1 specialization/compilation profiles pass: both CUDA
configurations on SM80 and the HIP configuration on gfx942, each with singleton,
ragged/transposed B plus activation, and K=0 cases. This is not GPU execution or
full native launcher replay. Evidence is in the task delivery packet; independent
acceptance is owned by the batch integrator.

Historical Contract/Correct/FPEquiv contents are retained under
`Proofs/Historical*.lean`; former host/Tensor context declarations moved unchanged
to `Proofs/LegacyContext.lean`. The existing host/autotuner proofs remain reusable
historical results, not the scope of the current public endpoints. Historical
numerical-port records keep their original scope; they do not describe the new
concrete scalar model or create a new primitive experiment queue.

The primary mathematical/native interfaces take `Kernel.Geometry` (regions,
dimensions, strides, activation and configuration) and any complete scheduling
permutation. They expose independent logical input/output functions rather than
raw interpreter states. `Proofs/AccessSafety.lean` follows the actual loop,
clamped masked pointers and FP16 stores; `Proofs/Interface.lean` identifies the
shared schedule with the original grid runner and reuses its established result.
The former raw endpoints remain in `Proofs/RawReal.lean` and `RawNative.lean`.
`Misc/TensorBridge.lean` retains the previous full Tensor signatures, native
integer/address safety and physical FP16 byte observations as ordinary theorems.
All Python, source quotation, contracts and TensorBridge files remain unchanged.

Current IO-format validation uses the isolated
`.veritile-cache/public-io-format-20261020/workspace_lean.py` overlay and
`batch13-autotune-*.log` logs. Both public entries have strict statement
allowlists, independent-formula checks and standard-base axiom checks. This
format pass does not rerun historical CPU/compiler evidence or GPU experiments.

## Implementation-pair review (2026-10-21)

`EquivIO.lean` binds the two existing actual source bodies to the same FP32 scalar and fixed MMA/MFMA instruction interpretation, FP16 stores, geometry and complete grid permutation. `Proofs/Equivalence.lean` reuses `programs_refinement` to construct both successful runs and prove equality of all typed memory cells. The only source changes are safe integer address/control expressions. The pair uses an empty experimental rule set, with no new numerical experiment. The stronger all-memory pair result also admits input/output aliases; the native formula theorem retains its original output-separation premises.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/NativeCorrect.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
