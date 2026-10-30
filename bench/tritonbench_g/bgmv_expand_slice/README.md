# bgmv_expand_slice

**Full-source conditional FP contract; native admission remains open.** The
corrected source preserves the product's reduction dtype and specializes output
storage before arithmetic promotion. The public `FPEquiv.lean` contract now
states whole-execution equality for all 72 modeled input/weight/cast/output
configurations, under an explicit native scalar addition law and a finite
operand domain. This is a correction of the formerly false fp32-only theorem,
not a claim that existing experiments have validated every native dtype.

`Proofs/NativeFP.lean` extracts the actual addition under the final store
conversion. Its domain checks the actual typed operands, in their actual
compute context, only on the selected non-sentinel branch, reached output-loop
iterations and accumulation arm. `requirements_present` checks all 72
configurations, so a missed source site cannot silently erase a guard. The
scalar premise requests commutativity only for the `(compute precision, dtype)`
selected by that source configuration. It contains no kernel execution,
candidate, output equation, cast law, product law or reduction law.
`output_first_execution` lifts this local scalar law through the complete
source, preserving casts, reduction order, pointers, masks, control flow,
registers, memory and failure. The new scalar premises are hypotheses; no
experiment admission or axiom is manufactured.

`Proofs/SupportedFP32.lean` retains the separately admitted fp32-output theorem
for every input/weight/cast choice and both masking and accumulation flags.
`NativeFP.fp32_native_of_rules` explicitly derives that source's required
scalar premise from the existing `ScalarArithmetic.Rules`. The domain still
checks the actual reduction/output operands. `Proofs/FP32Execution.lean`
supplies a successful nonempty execution witness with fp32 input cast to bf16
weights, fp32 output, a masked K tail, two output-loop iterations and signed
sentinel skip.

`Proofs/GeneralCountermodel.lean` preserves the historical fp32-only contract
and its old locator. For fp16 input/weights/output, K=N=BK=BN=SPLIT_N=1,
unit strides, active slot 0 and accumulation enabled, both complete kernels
execute. A model changing only fp16 addition to left projection satisfies
every selected fp32 scalar rule; the original stores 1 at Y[0], while the
candidate stores 2. The historical locator returns an empty domain.
`not_output_first_equiv` refutes that historical equivalence for every
`ScalarArithmetic.Rules` parameter. The corrected locator is nonempty on the
same source, and `not_corrected_native_law` proves this backend violates the
new scalar premise. This is a logical countermodel to inadequate assumptions,
not an IEEE or GPU numerical discrepancy.

`Proofs/AdditionProfiles.lean` supplies typed admission plumbing for the
existing bf16, bf16_fp32 and fp32 ADD-COMMUTE report rows. Validation remains an
explicit premise for the exact bound fragments. The bf16 experiments widen
their inputs to fp32 and perform explicit rounding conversions; these report
rows therefore do not by themselves establish the native bf16 primitive's
law used by BGMV. There is no supplied fp16 report. Other source-selected precision/dtype profiles, including the explicit fp64
output computation, also remain scalar premises unless separately validated.
The conditional full-source proof does not remove these concrete admission/lowering obligations and must
not be counted as unconditional hardware certification.

The three public entries share the complete source of
`bgmv_expand_slice.py`'s `_bgmv_expand_slice_kernel`:

- `Kernels.lean`: original and candidate that swaps the reduction result and
  loaded old output in the final addition. All input dtypes (fp16/bf16/fp32),
  both weight dtypes, explicit output dtype and both CAST_TYPE choices are
  represented, along with both EVEN_K and ADD_INPUTS branches.
- `Correct.lean`: both precision-erased sources implement an independent full
  rank-K dot product, plus **initial output** when accumulating. The contract
  includes successful flat-memory execution, actual access bounds and full-cell
  preservation. Initial vector, weights and old output are separate input
  values; the output allocation is listed once.
- `FPEquiv.lean`: the complete conditional native-addition proof, with the
  precise dtype-specific scalar premise and reached-operand domain described
  above. The public theorem no longer asserts the false fp32-only contract.

The real contract states K ≤ BK, positive BN, output stride and split count,
separate weight/output regions, disjoint allocations, initial input pins and
zero undef. Signed metadata is either -1 (skip) or nonnegative. Shapes and
strides remain symbolic. The input vector is loaded even at zero output length
on an active path, so its bounds are independent of output-loop count. The
source has no global N mask: all computed split rows need storage. Its output
base is `batch*outputBatchStride + split*length + slice*outputStride`; the split
term has no output-stride multiplier.

The source macro stages CAST_TYPE before register typing, so a dtype-changing
cast is retained. Output dtype is also specialized. The source's implicit
output-pointer conversion is explicit at each branch's accumulator assignment,
with no arithmetic between that conversion and the common store. This gives
both branches the actual stored dtype without widening the sum itself.
The Python early return is a non-sentinel active branch. These translation
choices have full-source real correspondence; the refreshed source regression
checks the corrected typed sum, promotion and conversion placement.

`Proofs/RealIOExecution.lean` instantiates the complete contract with four
concrete disjoint allocations, nonzero program IDs and slot, N=5/SPLIT_N=2,
K=3/BK=4, BN=2, nonunit strides, negative input, nonzero input padding and a
masked output tail. All dtype/cast/load choices and both versions produce
141/172/203 when accumulating and 30/58/86 otherwise. Only flat cells 91/94/97
may change. Masked weight padding is outside the weight allocation; the actual
active accesses remain in bounds. The signed sentinel preserves all memory.

`Proofs/TypedExecution.lean` uses the actual typed-sum backend and updated
guard path for fp32 inputs cast to bf16 weights and fp16 output. It provides
successful executions, inhabits both the finite domain and the scalar-law
premise in a consistent abstract model, and applies the full-source theorem.
`Proofs/NativeHalfExecution.lean` supplies the corresponding successful
fp16-weight/fp16-output path, whose final addition is actually native fp16.
Both are semantic consistency witnesses, not GPU numerical experiments.
`Proofs/RealFoundation.lean` contains the generic tile/mask/pointer lemmas
used by the current proof. `RealSetup` executes the source's full signed
base-offset sum and proves it equals the displayed read window on the
nonnegative active slots required by the public contract.
`Proofs/LegacyClippedOffset.lean` retains the historical explicit AST and its
proofs separately. That AST clipped the signed slot product before adding the
split offset, which disagrees with the source for some negative non-sentinel
slots. Its checked counterexample uses `3*(-2)+10`: old address 10, source
address 4. Current correctness proofs do not import the historical model.

Host launch composition, fixed-width address overflow and concrete IEEE
identities are outside these per-program contracts. Validation status is
recorded in `../verification_round.md`.
