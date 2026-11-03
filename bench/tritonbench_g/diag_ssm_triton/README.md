# diag_ssm_triton

Current repaired Python implementation: [diag_ssm_triton.py](diag_ssm_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Forward time recurrence and reverse carried/input/initial/transition gradients for FP32-real and complex64 mathematical interpretations.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.

The four repaired `@triton.jit` bodies and their four existing integer index
candidates are pasted into [Sources/Body.lean](Sources/Body.lean). Invocation
signatures bind runtime scalar arguments outside the quotations. The three
`tl.int64` shape casts, signed time loops, reverse-time branch, complex
conjugation signs, captured loads, masks and simultaneous tuple update remain
in the source. [jit-correspondence.json](jit-correspondence.json) checks all
eight bodies and their call metadata.

The [Python candidate](diag_ssm_triton_index_commuted.py) materializes the
existing column-index product commutation; its exact relationship and hashes
are recorded in [candidate-provenance.json](candidate-provenance.json). No
floating operation is reordered. The selected repaired host accepts float32
and complex64. The pre-existing fp16/bf16 mathematical invocation APIs are
retained without claiming those host launches.

The Real proofs connect the actual signed program to independent recurrences
and preserve every output and the raw-cell frame. Internal source-only cast
registers are retained; natural-coordinate mathematical proof models do not
assert equality of those internal states. Source matching, local checks and
independent proof replay are tracked separately in the validation records.
