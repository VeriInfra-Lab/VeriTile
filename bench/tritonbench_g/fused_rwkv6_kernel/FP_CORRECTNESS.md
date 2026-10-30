# Finite-precision stage correctness

`FPCorrect.lean` provides independent value specifications for the accepted
state-update and output slices under explicit fp32 storage. It also provides
a conditional final-step stage: compute the actual state update, then write
that result only when `STORE_FINAL_STATE` is enabled. The latter does not
load a precomputed final answer or assume that such an answer is correct.

Every input conversion, multiplication, addition, exp operation, reduction
and store conversion is retained. State results specify the complete masked
memory update; output results give exact active-lane readback and unchanged
memory elsewhere. Masked reduction terms are not discarded using real-number
zero or reassociation laws.

The state specification is the ordered operation graph
`store_cast(add(mul(previous, exp(cast(w))), mul(cast(k), cast(v))))`.
The output uses the pre-update state and the original source order
`sum((previous + (cast(k) * cast(v)) * cast(u)) * (cast(q) * scale))`,
with a store conversion after the reduction. Each operation is interpreted
by the selected backend.

The `fp32WithExp` profile makes add/mul/casts concrete ScalarOps operations
on binary32 bit patterns and isolates `exp32 : FP.Value .fp32 → FP.Value
.fp32` as the remaining primitive. These two state declarations are still
backend-value contracts. They do not constitute an implementation or proof
of exp, and they do not use `Real.exp` or an algebraic exp identity.

The output stage does not evaluate exp and has a complete software bit
instance. It uses scale=1, ascending key-index reduction, a positive-zero
initial accumulator and rounded fp32 addition at each reduction step. Its
output formula is independent of unused `rest` callbacks. A nonempty witness
has one valid output and one masked tail: a `2^-25` term is lost when added
to `1`, the actual stored result is `1`, and the tail remains unchanged.

A native binding must establish fp32 storage/address representation,
scalar cast/add/mul configuration, FMA contraction behavior, the exact exp
instruction/library implementation and its special-value handling, and the
actual reduction tree/order. The software reduction is not claimed to be
the generated GPU tree. The generic backend output allows arbitrary scale;
the closed software example selects one. The stage carry remains a supplied
materialized input. Complete recurrent loop composition, initial carry
construction, host-side `o.sum(0)`, autograd, reverse traversal and other
storage dtype profiles are outside these contracts. The conditional final
stage describes the actual last-step specialization when its input is the
carry before that step; it is not a full-loop final-state theorem.
