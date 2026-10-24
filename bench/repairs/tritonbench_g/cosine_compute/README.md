# cosine_compute: checked FP32 kernel and index variant

The independent formula is `out[i] = Real.cos(input[i])`. The production profile
is FP32. The complete readable DSL in `Kernels.lean` retains the original masked
load, explicit conversion to FP32, `tl.cos`, and FP32 store. Its existing
candidate changes only `program_id(0) * BLOCK_SIZE` to
`BLOCK_SIZE * program_id(0)`. Both public roots use these canonical sources.

Acceptance covers the kernel. Python supplies input assertions; allocator,
launch, device-context, and returned-Tensor correctness are outside the required
public proof. Earlier whole-wrapper proofs remain optional in
`Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`, and their existing helpers.
The original benchmark Python and original defect proofs are preserved.

## Contract and memory meaning

The input is a strided contiguous FP32 CUDA Tensor with no unresolved negative
view bit and fewer than `2**31` elements. Nonempty input pointers must be
four-byte aligned; empty inputs do not query the pointer. Scalar tensors,
aligned offset views, unusual singleton strides, NaN/Inf, signed zeros and
subnormals are admitted. There is no finite-input or cosine-accuracy-range guard.

`Contract.lean` and `TensorBridge.python_checks_iff_preconditions` establish exact
Python guard correspondence. The unchanged repaired wrapper uses `empty_like`,
returns it when empty, and otherwise launches BLOCK256 with exact integer
ceiling division on the input device. No copies, view resolution or additional
dtypes are introduced.

`IO.lean` owns the canonical record; `Correct.cosineIO` is its transparent
alias and declares `pid*256 + lane` input/output windows and the actual
shared `offset < n` mask. Its flat-memory contract quantifies over disjoint
allocations covering active lanes. The kernel must terminate, produce each
active value, and preserve every other cell. Inactive tail lanes require no
padding allocation. Tensor storage/offset bindings and whole-host results remain
available as optional helpers; the public claim starts from kernel memory and
program-ID conditions.

## Counterexamples and bounds

The original empty-input block calculation gives zero and then divides by zero;
the repaired empty return avoids it. At `n = 2**31`, the annotated signed-int32
extent becomes negative and every mask is false. The existing
`CosineNative.HostMath.zero_fails` and `all_masks_false_at_two31` results retain
these counterexamples. The revised count guard establishes correct signed
indices and masks for the fixed-block grid.

A stride-two logical input over `[0,1,2,3]` is `[0,2]`, but the original linear
loads use `[0,1]`. CPU replay retains that witness. Unresolved negative views are
rejected because raw storage need not equal logical values; no opaque FP32
cosine-evenness law is assumed for sign-changed words. CPU byte-offset FP32 views
with remainders 1, 2, or 3 motivate explicit alignment checks. Their CUDA
construction has not been tested.

## Public proofs and numerical scope

- `Correct.cosine_kernel_correctness` proves the independent Real cosine,
  successful finite flat execution, active access safety and the exterior
  frame. `Proofs/Projection.lean` retains the explicit source cast and proves
  its identity only in the mathematical interpretation, reusing the valid
  existing cosine result. The FP source is never replaced by this projection.
- `FPEquiv.index_commuted_equiv` proves that both actual FP32 sources execute
  successfully and have equal complete typed memory from arbitrary initial
  states. Integer multiplication commutes; the opaque FP32 cosine, cast and
  every memory operand remain unchanged.
- Public source-identity theorems bind both FP programs to `Kernels.lean`.
  The domain and numerical assumption lists are empty. Arbitrary active and
  inactive nonfinite values need no extra premise for this index-only change.

`#print_fp_assumptions index_commuted_equiv` reports **none**. There is no pending
numerical transformation experiment. The previous `LocalCosAccuracy` theorem is
retained as an optional hypothetical result in `Misc/HostFPEquiv.lean`; its
native-to-Real accuracy premise is no longer a public acceptance requirement
or experiment TODO. No native cosine accuracy, correct-rounding, IEEE
conformance, or general compiler/GPU correctness is claimed.

## Validation

Targeted Lean builds use `/tmp/veritile-lean-toolchain` with
`LEAN_NUM_THREADS=2`, including public standard-base axiom audits.

`python bench/repairs/tritonbench_g/cosine_compute/check_contract.py` checks exact
original AST preservation, the candidate's sole index-product swap, Python/Lean
guards, and both JIT bodies over CPU backing storage. Each variant passes ten
rejection and sixteen admission cases, with scalar/empty/tail/multiblock shapes,
offsets, unusual singleton strides, nonfinite inputs, inactive NaN loads, backing preservation and
original empty/signed-boundary/layout counterexamples.

CUDA allocation and launch are mocked. No GPU execution or new compiler
experiment was performed. CPU cosine checks indexing and storage, not GPU
numerical accuracy.

`Misc/TensorBridge.lean` contains the optional Python-check correspondence. The
primary `Correct.lean` and `FPEquiv.lean` entries retain the kernel results and
do not import the bridge.

`Misc/SourceBinding.lean` retains the source-identity helpers under their existing `FPEquiv` names.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.
