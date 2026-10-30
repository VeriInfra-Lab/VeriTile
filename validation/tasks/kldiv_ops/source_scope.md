# Source obligations for full-loop verification

The Python forward kernel first casts program id to int64, applies three
independent row strides, then iterates over all column blocks. With reduction
zero it performs masked elementwise stores. Every nonzero reduction mode
accumulates block sums in `loss_sum`, initialized to `0.0`, and performs one
unmasked scalar store after the loop. Host sum/mean/batchmean postprocessing
is separate from this kernel result.

The two formula branches must retain their actual expression trees:
`target * (log(maximum(target, eps)) - prediction)` and
`exp(target) * (target - prediction)`. Both masked loads explicitly use zero.
The reduction includes all lanes of `loss`, so an IEEE interpretation cannot
discard inactive-lane arithmetic merely because the loads were masked. The
runtime dtype of `eps` must be retained in source-profile checks.

The backward kernel is also a complete signed loop. It retains the offsets
and mask assignments before the loop as well as their per-iteration updates,
and stores either `target * -1` or `-exp(target)` under the mask. The wrapper's
later multiplication by `grad_output` is not part of this kernel.

The new proof surfaces include both flag branches, all reduction branches,
multiple blocks, source precision and conversions, actual access safety,
successful region/flat IO with an independent formula and full memory frame,
and complete original/candidate FP execution equivalence. Existing theorem
statements and docstrings remain untouched. This file records source review,
not a completed migration or a new validation result.

Pinned [Triton v3.5.1 math frontend](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/math.py)
accepts only fp32/fp64 tensors at `exp` and `log`. Thus the raw-target `exp`
branches require fp32/fp64 target storage in a source-facing floating family.
The default forward branch must instead check the dtype after `maximum` with
the runtime epsilon. This is a frontend-source inference, not a GPU compile
observation; a Lean automatic widening must not stand in for a missing Python cast.

For the source test's Python-float epsilon (`1e-6`), the pinned
[JIT specialization implementation](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/runtime/jit.py)
assigns a runtime fp32 argument (`create_specialize_impl`, float branch).
It is not a constexpr literal. In the pinned
[binary-operation frontend](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/semantic.py),
the scalar exemption applies to Python numbers, not runtime tensor values;
`maximum` therefore promotes a narrower floating target to fp32 with this
epsilon. An fp64 target instead promotes epsilon to fp64. Consequently the
default forward logarithm accepts all six floating target storage labels at
this frontend boundary, while the raw-target exponential branch still accepts
only fp32/fp64 targets. This inference covers a Python-float epsilon argument;
integer, constexpr or tensor epsilon calls need their own signature checks.
The typed Lean source encodes the fp32 runtime argument explicitly, as in the
existing `tests/lean/FloatRuntimeScalar.lean` regression.
`Proofs/ForwardPrecision.lean` checks independently written AST templates for
all 216 formal default-forward and 72 formal log-forward storage combinations,
in both variants. The checks retain both masked storage loads, the complete
maximum/log or exp tree, subtraction and multiplication, and each operand's
local precision before promotion. In particular, an fp64 target retains the
fp32 epsilon cast inside the fp64 clamp, while an fp64 prediction with a narrow
target retains the clamp/log computation at fp32. These source checks do not
certify native lowering or IEEE results.

The accumulation loop adds a separate frontend restriction. In pinned
[code generation](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/compiler/code_generator.py),
ordinary assignment converts `loss_sum = 0.0` to a tensor, and the loop dry run
requires each carried tensor to retain its initial type. The
[scalar conversion and promotion rules](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/semantic.py)
make that initial tensor fp32, but a loss involving fp64 prediction or target
promotes the update to fp64;
[sum](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/standard.py)
does not narrow fp64. Hence the source analysis predicts rejection for those
nonzero-reduction signatures, including runtime-zero rows. This restriction
does not apply to elementwise mode or merely fp64 output storage. The `SourceDomain`
typed admission predicate retains it; silently promoting the initializer
would change the Python source. This is a compiler-source inference, not an
observed native compile result. The ideal-real algorithm theorem alone cannot
certify these source signatures.

The same independent precision gate checks the masked elementwise store, the
unmasked block sum and carried addition, and the one post-loop scalar store
with its exact output conversion. `Proofs/Precision.lean` connects admission
to that AST: every source-supported nonzero reduction has an fp32 sum/carry
update. The broader formal fp64-carry trees remain checked as formal ASTs,
without weakening the admission predicate or widening the zero initializer.

The default backward branch also requires a nonzero weak scalar, `target * -1`.
Pinned [semantic.py](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/semantic.py)
selects the tensor dtype for an integer weak scalar and calls `scalar_constant`
at that dtype. A nonzero value requires `get_<dtype.name>`; pinned
[ir.cc](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/src/ir.cc)
exposes these scalar builders for bf16/fp16/fp32/fp64, but only type builders
for fp8. Therefore the source-domain predicate excludes fp8 targets in default
backward mode (24 admitted target/output pairs out of 36 formal pairs).
This is a compiler-source inference, not a native compilation result. The
formal six-storage AST family remains available as an explicitly broader
model. Log-target backward still requires fp32/fp64 target storage.

## Current source/candidate and proof surfaces

The already present candidate rewrites only `pid * loss_stride` (forward) or
`pid * new_grads_stride` (backward) to the commuted signed product. It preserves
all storage annotations and floating expressions. Its complete FP equivalence
therefore requires no numerical identity. This structural rewrite should not
be presented as a numerical optimization or as a performance improvement.

The internal ideal-real reference stages the constexpr `log_target` flag before
building the AST. Both flag branches retain the original formula. This staging
matches the typed source family, makes dtype erasure explicit, and does not
change the full-loop theorem statements. The source projection covers every
formal storage tuple; public contracts separately state frontend admission.

The full contracts cover arbitrary positive block size, zero or positive row
length, partial final blocks and independent input/output strides. Elementwise
forward and backward require separate input/output region names because later
blocks reload from memory; the reduced forward kernel permits aliases because
its only output store follows all reads. Flat contracts require exactly the
active input/output addresses in bounds, a disjoint allocation of distinct
regions, and the flattening bridge's zero undefined-cell convention. Every
other raw flat cell is framed. Host wrapper postprocessing is not included in
these kernel statements.
