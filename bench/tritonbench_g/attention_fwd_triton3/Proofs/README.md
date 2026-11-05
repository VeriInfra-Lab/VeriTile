# attention_fwd_triton3

The selected repaired Python source is [attention_fwd_triton3.py](../attention_fwd_triton3.py).
[Kernels.lean](Support/KernelEntryAdapters.lean) binds the caller ABI to the complete pasted function bodies in
[Sources/Body.lean](../Kernels.lean). Both `_attn_fwd` and `_attn_fwd_inner` retain their original
statements: signed integer casts, helper call, tuple assignment, clamps, masks, loads,
NaN-alpha repair, conversions and padded stores. Caller dtypes and constexpr bindings
are outside the pasted bodies. The repaired wrapper admits FP32 Q/K/V/M/L/Out.

The [candidate Python](../attention_fwd_triton3.py) only caches the repeated key-tail mask within
one iteration. [candidate-provenance.json](../../../../validation/tasks/attention_fwd_triton3/candidate-provenance.json) records the source
hashes and exact transformation; [source correspondence inventory](../../../../validation/source-correspondence.json)
checks all four complete original/candidate functions. Mutation tests reject missing
casts, changed clamps, missing alpha repair and incorrect caller/helper bindings.

[Correct.lean](../RealCorrect.lean) proves the complete ordinary-grid Real specification on the
six finite input windows in [IO.lean](../IO.lean). Its result is an independent maximum and
weighted denominator/numerator sums, seeded by INIT or the entry carry. The mathematical
interpretation uses exact WithBot Real arithmetic; it retains the signed32 pointer-start
conversion in its data indices, including wrapping metadata outside Python admission.
Zero iterations preserve the seed. Executing an all-masked block is a separate case.
Continuation writes M/L/O; END writes M+log2(../L) and O/L while preserving L.
The theorem includes trace safety, termination, all active padded outputs and the exact
exterior frame. Its static conditions remain `Valid c`; no floating cast or load identity
is added as a premise. This Real interpretation makes no IEEE arithmetic claim.

[FPEquiv.lean](../FPEquiv.lean) compares the complete repaired source with the actual CSE
candidate. Both ordinary grids execute safely and produce equal native FP32 graphs on
six arbitrary decoded input arrays and the raw FP32 scale slot. It uses no numerical
rewrite assumption and adds no finite-value restriction. The byte/Tensor projection
uses the same source execution in `Proofs/NativeProjection.lean`.

The historical FP16 and reciprocal full quotations are removed. Pure mathematical
lemmas shared by other tasks remain; they are proof helpers rather than alternative
implementations. [source-retirement.json](../../../../validation/tasks/attention_fwd_triton3/source-retirement.json) identifies the removed
source certificates and their current replacements. The source is not asserted equal
to the former quotation that omitted casts and inlined the helper differently.

Source correspondence, local builds and independent replay are separate gates.
[Validation records](../../../../validation/body-migration-progress.json) remain authoritative
about pending work; successful compilation alone does not complete this task.
