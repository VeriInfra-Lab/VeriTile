# Example specification migration

The reference is `TritonBenchVectorAddition/Correct.lean` and
`TritonBenchVectorAddition/FPEquiv.lean`.

Each completed case has a real correctness file and an independent FP equivalence file.
The correctness statement is `Spec.Real (io ⊨ mathematical_formula)`.
The FP statement is `originalKernel … ≡[R] optimizedKernel …` (or the corresponding
IO contracts for transformations with private scratch), with
`#print_fp_assumptions` separating the empirical atoms and strict certificates
actually used by its proof. Both proof files import the shared sources in `Kernels.lean`. FP execution
and contract helpers live in the example's `Proofs/FP.lean` and never import the
correctness counterpart. See the [example index](../bench/examples/README.md)
for the shared implementations and precise observation scopes.

All 19 examples keep IO definitions, necessary conditions and specifications in
their reader entries. Auxiliary real lemmas live in `Proofs/Real.lean`; FP
execution, recurrence and scalar composition lemmas live in `Proofs/FP.lean`
when needed. Public FP contracts expose precision, operand domains, schedules
and comparison requirements. The entry proofs compose the checked lemmas;
helper modules do not import the entry files.

Experimental shape and distribution select assumptions. They are not extra
shape conditions on the subsequent Lean derivation. Kernel dimensions remain
parameters except where a case was already explicitly a fixed-rank slice.

## Coverage ledger

This table tracks the original cases, not just newly compiling files. A real
proof or an unrelated local rewrite does not complete an existing algorithmic
FP equivalence. Pending entries must not be advertised as proved.

| Original case | Correctness file | FP transformation and status |
|---|---|---|
| TritonBench vector addition | `TritonBenchVectorAddition/Correct.lean` — checked for original and optimized sources | `TritonBenchVectorAddition/FPEquiv.lean` — checked; add commutation |
| Aligned vector addition | `VectorAdd/Correct.lean` — checked for original and optimized sources | `VectorAdd/FPEquiv.lean` — checked; add commutation |
| Masked vector addition | `FlatVectorAdd/Correct.lean` — checked for original and optimized sources | `FlatVectorAdd/FPEquiv.lean` — checked; add commutation |
| Float dtype addition | `FloatDTypeAdd/Correct.lean` — checked for original and optimized sources, including empty tiles | `FloatDTypeAdd/FPEquiv.lean` — checked; add commutation; output cast retained |
| Row-wise sum | `RowWiseSum/Correct.lean` — checked for original and optimized sources | `RowWiseSum/FPEquiv.lean` — strict fp32 ADD-COMMUTE with empirical ADD-ASSOC/ADD-ZERO; dimensions and reduction schedules remain symbolic |
| Row-wise max | `RowWiseMax/Correct.lean` — checked for original and optimized sources | `RowWiseMax/FPEquiv.lean` — checked; inline the load and reduction into the store, preserving the same reduction and input order; no numerical assumptions |
| Online softmax | `OnlineSoftmax/Correct.lean` — checked for batch and complete online output; bounded flat-memory stores/frame for both sources | `OnlineSoftmax/FPEquiv.lean` — checked for actual stored outputs on both sides; libdevice EXP-SUB and scalar arithmetic, symbolic positive row length and memory frames |
| mHC depth | `HyperConnectionsDepth/Correct.lean` — checked for general matrix sources and finite Sinkhorn iterations, including bounded flat memory for both original and optimized versions | `HyperConnectionsDepth/FPEquiv.lean` — checked for symbolic dimensions and iteration count; final pointwise add commutation |
| mHC width | `HyperConnectionsWidth/Correct.lean` — checked for general matrix sources and finite Sinkhorn iterations, including bounded flat memory for both original and optimized versions | `HyperConnectionsWidth/FPEquiv.lean` — strict scalar multiplication commutation; matrix logit division versus reciprocal multiplication uses empirical DIV-MUL-RCP, with symbolic dimensions, iteration count and unchanged matrix products |
| Adam-named Lion update | `AdamUpdateGridLaunch/Correct.lean` — checked for both sources per program; original grid proofs retained | `AdamUpdateGridLaunch/FPEquiv.lean` — checked per program; momentum addition commutation, masked in-place stores retained |
| Stable softmax | `SoftmaxStable/Correct.lean` — checked for both original kernels against the softmax formula | `SoftmaxStable/FPEquiv.lean` — checked for the libdevice.exp kernels, using admitted scalar arithmetic and EXP-SUB; symbolic row length, scheduled sums, bf16 stores and frames retained |
| Log-exp elimination | `LogExp/Correct.lean` — checked for the fixed reference and guarded candidate against the identity formula | `LogExp/FPEquiv.lean` — checked using the admitted conditional `log_exp_cancel .libdevice` atom |
| Stable logsumexp | `StableLogSumExp/Correct.lean` — checked for the direct reference and conditional candidate against logsumexp; the historical shift's real proof is in `Proofs/LegacyCorrect.lean` | `StableLogSumExp/FPEquiv.lean` — `logsumexp_equiv` checks the conditional tl.log candidate against the unchanged direct reference, using `log_mul_split .tl`, `log_exp_cancel .tl`, exp-sub and arithmetic. The older general FP target is refuted by a singleton logical countermodel in `Proofs/Unconditional.lean`. |
| Softmax reciprocal | `SoftmaxReciprocal/Correct.lean` — checked for both original kernels against the softmax formula | `SoftmaxReciprocal/FPEquiv.lean` — ordinary fp32 division versus a shared reciprocal, with the original bf16 output cast and explicit finite/nonzero operand domain |
| Float dtype softmax | `FloatDTypeSoftmax/Correct.lean` — checked for both original fp32-load/fp64-work kernels against the softmax formula | `FloatDTypeSoftmax/FPEquiv.lean` — fp32 load, fp64 work, fp32 output; only the casted division/reciprocal relation is assumed |
| Fused SiLU | `FusedSiLU/Correct.lean` — checked for both original kernels against residual + silu(x · gate), including empty blocks and scratch framing | `FusedSiLU/FPEquiv.lean` — checked; original fused versus materialized pipeline, with no numerical assumptions |
| Fused SwiGLU | `FusedSwiglu/Correct.lean` — checked for both original kernels against silu(x) · y, including empty blocks, tail masks and scratch framing | `FusedSwiglu/FPEquiv.lean` — checked; original fused versus materialized pipeline, with bf16 casts, tail masks and no numerical assumptions |
| Welford | `Welford/Correct.lean` — checked for both original kernels against population mean and variance; both output windows and memory framing | `Welford/FPEquiv.lean` — checked under scalar arithmetic and two strict bounded-count certificates, for `0 < N <= 2^24`; both original bf16 outputs and memory frames retained |
| Fused layernorm | `FusedLayerNorm/Correct.lean` — checked for both original kernels against population-variance normalization and affine transformation | `FusedLayerNorm/FPEquiv.lean` — checked for `N <= 2^24`, including empty output rows; original statistics, affine suffix, bf16 stores and frames retained |

There are 19 correctness modules and 19 modules with completed FP specifications.
They expose 21 public FP specifications. The [current rule-usage guide](./FPRemainingAdmissionGaps.md)
records the group/specification split and the 12 empirical instances actually used.
StableLogSumExp proves the conditional candidate. Its older general FP target
is refuted in `StableLogSumExp/Proofs/Unconditional.lean` by a logical model of
all selected atoms, with different stored outputs on a legal singleton row.
This is not a GPU or two-gates failure; the fallback version is the supported
optimized source. The eight `Proofs/LegacyRealEquiv.lean` modules
retain proofs with real intermediate arithmetic and their stated cast semantics.
Their presence does not complete a pending FP transformation.

The nine scalar/reduction example pairs now each expose an optimized real
specification in addition to the original one. The statements reference the
optimized source directly and prove the same mathematical formula, with the
IO bounds, active-lane masks and memory frames retained. They do not use the
FP equivalence theorem or any experiment-selected atom. Reversed row sum uses
a bijection on `Fin B`; inlined row max is proved by direct execution. The
optimized aligned addition additionally covers empty tiles. The optimized
Lion statement covers each program's two masked in-place outputs; the existing
whole-grid launch theorems remain stated about the original implementation.

The SiLU and SwiGLU materialized kernels retain the original `ComputeKernel.seq`
scope: one concatenation of stage bodies. Both real and FP specifications explicitly
declare scratch windows and prove that every cell outside output and scratch
windows is preserved. They do not claim a new separate-launch theorem.

Welford and LayerNorm retain symbolic row length and stride. The strict FP count certificates require `N <= 2^24`;
Welford requires a nonempty row because its empty mean divides by zero. These are
per-program specifications: the original Welford sources write each scalar
output at offset zero, so this does not assert a race-free multi-program
Welford launch. Both real correctness proofs also cover zero-length rows under Lean's total
real arithmetic. For Welford, that means zero mean and variance; for LayerNorm,
there are no output lanes. The new `KernelIO₁ₓ₂.Implements` interface requires
both results and frames outside the union of the two output windows and any
declared scratch windows.

## Admission and proof boundaries

The main, supplemental and bounded-count reports retain historical precision
instances. Experimental admission uses a local-ULP mean-bias budget (`tau=0.05`,
five SEs) and a peak absolute-error ratio gate; z is diagnostic. Only dual-PASS
rows enter the generated historical tables. [CurrentAdmission](../VeriTile/Triton/Float/CurrentAdmission.lean)
separately removes matching strict certificates from the current experimental
pool. No table contains a whole softmax, logsumexp, normalization, reduction or
recurrence atom.

The supplemental EXP-SUB implementation uses libdevice.exp. Its bf16,
bf16-input/fp32-work/bf16-output and fp32 instances are admitted with the
configured magnitude PASS threshold of 10. Its identity is
part of the report contract. The configured tl.exp experiment rejected that intrinsic version
(B=0.1608954387 ULP > 0.05). The stable/online softmax and logsumexp examples
therefore explicitly use libdevice.exp in both their Correct and FP sources,
with separate AST and opaque FP symbols for the two implementations. LOG-MUL domain events and unsupported fp64 combinations
remain unaccepted. DIV-RCP uses div_rn; DIV-MUL-RCP tests ordinary division.

The two reciprocal examples use `Guarded.IO`: the signature includes a domain
contract checking finite exponential values and a finite, nonzero denominator
at the shared prefix. They quantify over opaque numerical interpretations
satisfying the selected scalar theory, prove both runs succeed, and frame every
cell outside the output window. Exp, max and sum are identical opaque operations
on both sides; no whole-softmax law or positivity of abstract exp is assumed.
Only the fp64 relation's final fp32 outputs are equated. The fp32 example lifts
its admitted fp32 relation through the common bf16 output cast.

`ComputeDType.fp64` and the DSL preserve explicit float64 casts and precision
through arithmetic, max, sum, exp and log in these examples. The wide FP example
binds its fp32 load before widening so each precision boundary is explicit.
Binary64 constant projection is partial (finite normals); the structural
interpreter deliberately rejects raw fp64 payload constants and typed fp64
loads, which these examples do not use. This is not a complete IEEE evaluator.

[The current proof boundaries](./FPRemainingAdmissionGaps.md) distinguish actual
dependencies from available catalogue entries. Historical main-table-only
countermodels do not establish a gap in the combined strict and empirical theory.
The historical unconditional logsumexp shift's general FP specification is refuted
under the currently selected atomic theory by the `refutation` declarations in
[Unconditional.lean](../bench/examples/StableLogSumExp/Proofs/Unconditional.lean).
The singleton countermodel satisfies all selected rules and domain checks,
while the kernels store different outputs. This is a logical disproof under
the abstract theory, not a GPU or two-gates failure.

The selected LogExp API admits guarded log-exp and log-product candidates,
with separate tl.log and libdevice.log evidence. The accepted log-exp candidates
use libdevice.exp; the measured tl.exp candidates fail the configured bias gate.
Guarded log-exp returns `a` only for `0.5 < abs(a) <= 80`, otherwise retaining
its reference. Guarded log-product keeps the product log in `[0.5, 2]` and splits
outside it. Both fallbacks and inactive-argument masks remain in the candidate.
These admissions do not enable the API's unconditional candidates or a different
backend/precision relation. StableLogSumExp uses the two guarded tl.log variants
without changing its direct reference. Its scheduled IO contract requires fp32
comparison support, not an assumed branch outcome. The frozen measurements are
in the [log report](../experiments/floating_point/supplement/log_report/summary.md).

The current `Spec.Derivation` supports empirical atoms, fixed-theory strict
certificates, symmetry, transitivity and common sequential context. A
`ProgramSyntax` view selects its strict theory and may additionally enable independently
proved structural execution steps through `Spec.ProgramDerivation`, without
changing the public `≡[R]` notation or the existing syntax-only views. Each step
preserves the public signature; syntax steps also preserve private scratch
metadata. Whole-program structural steps cannot be framed inside arbitrary
statement contexts.

`Float/Structural` interprets floating values with an arbitrary carrier and
arbitrary numerical functions, preserving dtype and compute-precision tags.
`Float/StructuralIO` requires both executions to succeed, output cells to agree,
and every cell outside each implementation's output and private scratch windows
to remain unchanged. Scratch cannot alias public inputs or output. The SiLU
proof retains the exact original kernels and derives the same opaque numerical
call tree by store/load forwarding. Its assumption printer reports `none`.
This is a structural theorem of the FP model, not an IEEE execution theorem or
a claim that a newly measured whole-kernel numerical test passed.

The SwiGLU FP proof uses the actual elaborated syntax: the intermediate load
annotation becomes a bf16-typed load, with no additional cast. Its legacy
proof needed rounding idempotence because that model rounded again on a typed
store. The structural model copies an already typed value on store and retains
all explicit casts as opaque operations. The new proof therefore also reports
`none`. Its `MaskedKernelIO₂` signature preserves the complete active-lane
predicate, and its frame preserves inactive output and scratch cells. It works
for arbitrary element count and block size, including zero and partial blocks;
inactive scratch values are not used as a forwarding premise.

The structural evaluator currently supports straight-line assignments, typed
loads/stores, masks and a subset of expressions. Max reduction is interpreted
as an arbitrary operation retaining compute precision, the entire input tile,
shape, axis and `keepDims`; an empty reduction axis fails, as in the original
semantics. The row-wise max FP proof preserves that operation while eliminating
the `values` and `result` register bindings. Its stride and positive row length
are symbolic, with the same nonempty-row condition as its Correct counterpart.
Its one-input IO interface requires successful executions, equal output cells
and memory framing on each side.

The row-wise sum proof specializes the original mathematical kernel to fp32
input and accumulation and reverses the lane addresses before `tl.sum`. The
DSL preserves the input's fp32 compute annotation on the reduction and its
result. The mathematical projection still equals `RowWiseSum/Correct.lean`'s kernel.

`Float/Equational` supplies expression congruence and substitution of the exact
scalar ADD-COMMUTE and ADD-ASSOC templates. Its reduction-tree theorem derives
equivalence from a permutation of the leaves, without a whole-reduction atom
or a floating additive-identity assumption. `Float/TermModel` expands sum into
an arbitrary valid addition schedule. Each input occurs exactly once; any
explicit padding zeros remain leaves with their multiplicity preserved. The
schedule retains precision and layout and cannot depend on numerical inputs.
A concrete valid schedule exists even for empty rows.

The one-input IO view can now use these term derivations to relate actual
successful abstract executions, with the same typed-output and memory-frame
obligations as its structural steps. Other primitives stay opaque. The public
notation remains `lhs ≡[R] rhs`. The sum example prints strict `add_commute`
separately from empirical `add_assoc` and `add_zero`, with their fp32 domains.
This derives a theorem in the selected FP model; it neither
replays the GPU report nor claims an IEEE or whole-kernel statistical guarantee.

Unsupported syntax still fails explicitly. Counted loops and conditionals
now have execution lemmas. Welford also has scalar-derived loop and schedule
comparisons plus a scheduled IO contract with syntactic domain checks; its two
integer-count conversions now use strict bounded-count certificates. The
public Welford FP theorem retains `0 < N <= 2^24`. LayerNorm reuses the
unrounded statistics through its unchanged affine suffix and completes its
three-input FP contract for `N <= 2^24`, including empty output rows. Real ring identities
cannot be installed as structural FP rules.

Stable softmax has independent libdevice source definitions and a scheduled
one-input contract. `SoftmaxStableFPEquiv.softmax_stable_equiv` derives its
reduction and normalization from the admitted scalar arithmetic and libdevice
EXP-SUB atoms. Max and bf16 casts remain opaque; the row length is symbolic
and positive. `#print_fp_assumptions` separates strict certificates from the
empirical scalar dependencies.

`OnlineSoftmaxFPEquiv.online_softmax_equiv` compares `batch` with `online`
using `Float/ScheduledIO`. The complete online source runs the recurrence,
rereads the row, and stores `exp(x-m)/l`. Both kernels must execute successfully,
their stored output cells agree, and all other memory is framed. The signature
retains fp32, the row layout, symbolic size and actual intermediate domains.
The real output theorem proves the same two-pass source against the independent
softmax formula, with a `KernelIO₁` specification for bounded flat memory. The old register observations are auxiliary
lemmas, not the public equivalence scope.

Stable logsumexp retains its single bf16 store at `pid` and uses libdevice.exp.
The conditional candidate's log-product and log-exp atoms are admitted and
bound, and its complete FP specification preserves both fallbacks. The
historical unconditional shift's general FP target has a checked logical
counterexample in `StableLogSumExp/Proofs/Unconditional.lean`. Count-conversion
certificates retain their explicit integer bound. Any external unresolved
FP premise remains visible to the assumption printer, including equations
accessed through record fields.

Legacy `KernelIO₁.Equiv` proofs quantify over a boundary-rounding model. They
are not proofs under the new two-gates-selected atom calculus and do not count
as completed FP entries in this ledger.

For real correctness, explicit narrow-float storage annotations must also be
erased: merely setting rounding to the identity does not make a typed fp32 or
bf16 memory cell a real memory cell. This distinction matters for the
`KernelIO` output readback. FP proofs retain dtype annotations.

## Validation

The current strict-rule classification is in
[CurrentAdmission](../VeriTile/Triton/Float/CurrentAdmission.lean).
Proof builds, regression checks and independent replay validate the current
source tree. Experimental rules retain the reports required by their admission.

For code changes, run `lake build VeriTile TritonBenchSpecExamples` and the
relevant existing regressions. The example target includes all `bench.examples`
modules and the imported TritonBench vector-addition module. Check:

- Shared source bindings and independent Real/FP imports.
- Public formulas, precision, actual operand domains, masks and scratch windows.
- Successful execution, output readback, bounds and memory frames.
- `#print_fp_assumptions` on the actual public theorem, separating strict
  certificates, used empirical instances and unresolved external premises.

Run axiom/statement audits and the independent comparator for changed proofs.
Reuse matching strict certificates and frozen experimental evidence; a change
of relation, precision, backend or domain needs its own applicable evidence.
