# attn_fwd_triton

Current repaired Python implementation: [attn_fwd_triton.py](../attn_fwd_triton.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Causal base-2 attention: sum over n=0..m of exp(log(../2)*score(../m,n))*V(../n,d), divided by the sum of those weights. score is the complete headDim Q/K dot times the original packed query-block and key-block scales.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The helper and both original/candidate callers in `Kernels.lean` paste the complete Python function bodies. Pointer dtypes and argument bindings live in `CallSignature`; source casts, masks, both legal `STAGE` values, helper argument captures and tuple returns are retained. The second call assigns its final maximum to Python `_`, as in the source.

The candidate changes only the integer remainder expression to quotient/remainder arithmetic; its complete Python source and hash provenance are checked by `jit-correspondence.json` and `candidate-provenance.json`. `Proofs/ActualSource.lean` through `ActualFlatExecution.lean`, `FlatCausal.lean` and `FlatTraceSafety.lean` connect the actual source to the independent causal formula and full-grid memory safety. `source-proof-retirement.json` records the obsolete inline execution/source certificates removed during this migration.

Completion requires the exact current independent replay receipt recorded in `validation/body-migration-progress.json`; local builds and source comparison are separate gates.
