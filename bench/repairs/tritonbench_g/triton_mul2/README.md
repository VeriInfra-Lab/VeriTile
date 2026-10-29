# triton_mul2: kernel correctness and multiplication commutation

The independent target is `out[i] = 2 * entry_x[i]` on every active lane.
For the in-place kernel, `out` and `x` denote the same region: the formula
uses the input value before the kernel executes. Both source kernels and both
`2 * x` → `x * 2` candidates are covered. Input, arithmetic and output are
FP32, following the default for the benchmark's unspecified input dtype.
`Spec.Real` states the mathematical formula; it does not assert exact native
floating-point equality to a real product.

`Misc/TensorBridge.lean` provides the optional Python-check correspondence. The main mathematical and FP declarations remain in `Correct.lean` and `FPEquiv.lean`; neither imports the bridge.

## Source and proof organization

- `Kernels.lean` is the canonical readable, explicitly FP32 Triton DSL source.
  In-place kernels specialize the same body with equal input/output regions.
  `Correct.lean` and `FPEquiv.lean` reference these exact kernels.
- `Correct.lean` has two `MaskedKernelIO₁` specifications for the original
  separate-output and in-place kernels. `Misc/CandidateCorrect.lean` retains the two
  candidate correctness theorems. Each proves successful execution,
  active output values, masked access bounds and the exterior memory frame.
  The source's Real projection reuses the established original proofs through
  `Proofs/Projection.lean`; no arithmetic proof is duplicated.
- `FPEquiv.lean` proves both intended FP32 multiplication commutations with
  `≡[[]]`, using the shared strict theorem. Both
  full typed sources execute successfully and agree on every memory cell.
  No input/output region separation is required for the equivalence.
- `Contract.lean` defines input requirements; `Proofs/Contract.lean` proves
  exact successful Python admission equivalence and signed-index bounds.
  `Proofs/MemoryContract.lean` connects checked metadata to logical reads,
  actual data pointers, alignment and allocation bounds using the shared
  `VeriTile.Torch.FP32.TensorLaws` binding.
- `Proofs/Rewrite.lean` executes the actual sources with their FP32 tags.
  It proves memory equality using strict commutation only where the source store mask
  is true, and separately proves the original FP execution's exterior frame
  with no numerical premise. `Proofs/MaskedDomain.lean` proves empty-input,
  arbitrary inactive-load and one-element-tail domain properties.

Public proof entries do not execute the Python wrappers, allocate output,
model launch orchestration, or prove returned Tensor identity. Prior
`Proofs/Host.lean`, scheduling, native and Real-grid results remain optional
valid helpers; they are not imported by either public kernel proof entry.

## Input assertions and their memory meaning

`triton_mul2_repaired.py` retains both original Python kernel bodies and the
wrapper default `BLOCK_SIZE=16`. It also displays the two commuted candidate
kernels. The guards require:

- A strided FP32 CUDA Tensor, contiguous in the default format, with no
  unresolved negative view bit.
- `numel < 2**31`. For nonempty input, the actual data pointer is divisible by
  four. Empty tensors impose no unused pointer-alignment requirement.
- A Python `int`, excluding `bool`, for `BLOCK_SIZE`: a power of two from
  one through `2**20`, the installed Triton 3.7.1 frontend block-shape range.

Python guards and `InputContract` agree on successful admission. Scalars,
empty shapes, arbitrary rank and contiguous offset views are included;
NaNs and infinities are admitted. Inputs are passed directly, without copying,
negative-view resolution or general-stride adaptation. Device resource limits
and successful compilation remain runtime boundaries.

Contiguity implies that logical element `i` is at `data_ptr + 4*i`, including
nonzero storage offsets. The absent negative-view bit makes the raw loaded
word equal the logical Tensor value. `MemoryContract.direct_read`,
`actual_input_pointer` and `active_bytes` prove these connections under shared
Tensor API laws, with no kernel-result premise. See the references in
[`VeriTile/Torch/FP32.lean`](../../../../VeriTile/Torch/FP32.lean) and the shared
[physical load/store model](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts)
for Torch metadata semantics, byte bounds, natural alignment and masked
accesses. In particular, scalar FP32 accesses use four bytes; this does not
assert that arbitrary vectorized machine instructions need only four-byte
alignment.

The per-program mathematical and FP theorems quantify over abstract kernel
memory and program IDs, and can consequently be stronger than host admission.
`native_indices` connects an admitted `n`, block size and launched program to
the source's signed-int32 offsets and masks, including inactive tail lanes.
Kernel memory bounds are part of `MaskedKernelIO₁`; allocating or validating
an output Tensor is outside the wrapper-proof scope.

## Original-source limitations

The original benchmark Python is preserved in `bench/tritonbench_g/triton_mul2`.
Its wrapper lacks a contiguity precondition. For backing values
`[1,2,3,4,5,6,7,8]`, the stride-two view has logical values `[1,3,5,7]`.
The original in-place kernel doubles the first four physical cells, producing
logical values `[2,6,5,7]` instead of `[2,6,10,14]`, and changes backing cell 1,
which lies outside the logical view. The ordinary output kernel similarly
reads `[1,2,3,4]`, yielding `[2,4,6,8]` rather than the requested logical result.
The existing finite-index counterexample at `numel = 2**31` also remains:
signed-int32 extent conversion makes the extent negative and masks all lanes.

These are documented limitations of the original unrestricted interface.
This directory explicitly revises its input contract; it does not certify
those unsupported inputs or silently materialize them.

## Numerical rule and validation

The only transformation is strict FP32 multiplication commutation, proved by
`ScalarOpsLaws.mul_comm` and lifted through `ScalarArithmetic.mul_comm_strict`
into the fixed `Guarded.fp32Theory`. Its arithmetic profile is
`Software.strictFP32` with `.ieeeValue`. Both public statements use `≡[[]]`
and require no empirical `R`. The retained finite-operand domain
is attached to the actual multiplication site after the load prefix, only for
active stored lanes. Finiteness is not a Python guard. Inactive lanes may be
nonfinite, and an entirely masked program needs no finiteness premise.
Basic FP32 load, multiplication, store, rounding and literal semantics are
modeling boundaries, not extra native-to-IEEE numerical experiment tasks.

Historical numerical evidence is the accepted `fp32/MUL-COMMUTE` report in
`experiments/floating_point/report/summary.json`, exported by
`VeriTile/Triton/Float/ReportedAdmission.lean`: 4096 replicates, both gates
PASS, report SHA256
`c00ff0be806504ef7aadd1d85accf7c34303b538a953ffd0cf433dbea1c0e0d4`.
The report remains intact with its original scope. It is no longer a premise
of the strict commutation theorems, whose model interpretation stays explicit.
Independent replay was not rerun, and native GPU execution was not performed.

Targeted validation:

```sh
lake build bench.repairs.tritonbench_g.triton_mul2.Correct \
  bench.repairs.tritonbench_g.triton_mul2.FPEquiv \
  bench.repairs.tritonbench_g.triton_mul2.Misc.CandidateCorrect \
  bench.repairs.tritonbench_g.triton_mul2.Misc.SourceBinding \
  bench.repairs.tritonbench_g.triton_mul2.Proofs.MaskedDomain
python bench/repairs/tritonbench_g/triton_mul2/check_contract.py
```

The contract check retains the two original-kernel AST comparisons and checks
that each candidate changes only operand order. Existing CPU backing-storage
checks cover 132 cases and 34 guard rejections, plus empty inputs, singleton
and offset views, nonfinite admission and the original stride counterexample.
CUDA launches are mocked: these checks establish source/contract behavior,
not GPU numerics. Lean axiom audits require only the standard base; the FP
assumption audit separates strict multiplication commutation from the empty
empirical assumption list.

Public IO organization: `Correct.lean` retains the two original masked IO
contracts, including source identity, address windows, masks and in-place aliases.
`Misc/CandidateCorrect.lean` retains the candidate contracts and correctness results.
Both entries guard their statements with `#stmtSurfaceSubset`. `FPEquiv.lean`
retains the two shared floating-program equivalences. The auxiliary native
frame theorem `FPEquiv.original_memory_frame` now lives in
`Proofs/FPFacts.lean`; its statement and name are preserved.

`Misc/SourceBinding.lean` retains the source-identity helpers under their existing `FPEquiv` names.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/CandidateCorrect.lean`
- `Misc/SourceBinding.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
