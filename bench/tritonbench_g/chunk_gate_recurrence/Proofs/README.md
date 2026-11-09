# chunk_gate_recurrence

Current repaired Python implementation: [chunk_gate_recurrence.py](../chunk_gate_recurrence.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Independent exclusive forward and reverse suffix/product formulas; logical forward state or backward state, DG = sum over key:Fin64 and value:Fin16 of reverse*saved, and initial gradient. All 1024 contraction terms are retained.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The four complete `triton { ... }` bodies in `Kernels.lean` copy the repaired
Python `_fwd_recurrence` and `_bwd_recurrence` and the existing sum-left
candidate in `chunk_gate_recurrence_sum_left.py`. The candidate changes only
`acc * d_i + S_i` and `Dacc * d_i + DS_i` to their existing sum-left forms.
The original counted loops, four seed strides, destination-dependent casts,
DG reduction and final DL store are retained; there are no added `offs_k` or
`offs_v` source statements. `jit-correspondence.json` checks all four bodies.

Call signatures, runtime integer bindings, storage selections and nullable
seed presence stay outside the quotations. The nullable pointer is rebased
only in its presence arm; the optional frontend checks this and introduces a
fresh branch-local SSA binding. The independent reference retains the typed
FP32 seed read as well as its explicit cast. `SourceTypes.lean` contains only
storage metadata; `Sources/` now forwards to these same authoritative bodies.
Successful execution, backed accesses, output values, frames, independent Real
formulas and complete original/candidate FP execution are separate obligations.
Source comparison and local compilation do not establish final acceptance;
current-source independent replay is tracked in `validation/`.
