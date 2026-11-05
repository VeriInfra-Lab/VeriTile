# layer_norm_triton

Current repaired Python implementation: [layer_norm_triton.py](../layer_norm_triton.py).

[Pasted JIT bodies](../Kernels.lean) · [Lean entry points](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

All three repaired Python JIT bodies and all three existing scheduled candidate
bodies are copied in full into `Sources/Body.lean`. Call signatures carry FP32
storage, runtime scalar types and constexpr arguments outside the quotations.
The signed64 casts, grouped `while` and its final increment, load/cast operations,
reductions, masks, saved statistics and store order are retained. The original
and candidate Python files are unchanged.

`jit-correspondence.json` checks every complete function body. The candidate
provenance record and Python tests check the entire module AST against exactly
the existing three scheduling transformations, including input checks and host
forward/backward launches. This source check does not establish proof completion.

The grouped backward public FP entry requires a positive group stride, matching
the repaired host's fixed positive group sizes. `Proofs/SourceWhileTrace.lean`
connects the actual retained signed loop body to ordinary while execution under
that condition. Real correctness uses independent forward, DX and parameter
reduction formulas. FP comparison retains the actual typed loads, explicit
casts, integer-to-float conversion and complete output memory/frame.

Current migration completion is recorded in
`validation/body-migration-progress.json`; source compilation alone is not an
accepted independent proof replay.
