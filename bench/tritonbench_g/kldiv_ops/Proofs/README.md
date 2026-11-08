# kldiv_ops

Current repaired Python implementation: [kldiv_ops.py](../kldiv_ops.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

forward: pointwise KL loss when red=0, otherwise an unnormalized Fin cols row sum; only raw-target branch clamps to runtime epsilon. backward: -exp(../target) in log-target mode, otherwise -target.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` copies all four complete Python function bodies verbatim into
`triton { ... }`. The forward and backward originals come from `kldiv_ops.py`;
`kldiv_candidates.py` records the existing output-pointer multiplication
candidate, changing only `pid * stride` to `stride * pid`. Caller signatures,
storage profiles, scalar argument bindings and pointer offsets are outside the
bodies. Runtime epsilon remains a runtime FP32 input; all source integer casts,
unused initial backward offsets/mask, branches, loops and stores are retained.

The Real specification independently states pointwise KL loss or the full
unnormalized row sum and the unscaled prediction derivative. The source
execution and safety proofs cover every signed loop iteration, active access,
output and exterior memory cell. The FP comparison connects both complete
programs by signed integer pointer multiplication commutativity; it needs no
floating numerical rewrite. Ordered FP formulas retain actual input-load and
implicit storage conversions without adding casts to the Python source.

The retained mixed-storage backward profiles instantiate these same copied
bodies. Local build/source checks and an accepted current-source independent
proof replay are recorded separately in the validation progress and receipts.
