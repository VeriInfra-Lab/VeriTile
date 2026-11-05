# attention_kernel

Current repaired Python implementation: [attention_kernel.py](../attention_kernel.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Full-N base-two softmax weighted value sum with source-exact 1.44269504 coefficient and both bias windows.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` directly quotes both complete 27-statement Python bodies:
the repaired source and `attention_kernel_query_early.py`, the existing candidate
that moves the query load and scale before the other block-pointer setup.
`jit-correspondence.json` checks both independently. FP16/BF16 storage choices,
runtime strides and scale, and the `OUT_DTYPE` constexpr are invocation metadata
outside the quotations. The block-pointer shapes, singleton boundary tuples,
int64 casts, bias windows, both typed dots, tail mask and output cast remain.
`SourceTypes.lean` contains only the storage selection.

The execution reference retains the actual FP32 `qk` accumulator arithmetic and
its conversions, including the storage-typed bias coefficient. Native staging
keeps the corresponding half/BF16 literal conversion and widening to FP32;
there is no assumed equality between distinct opaque dtype operations. The
independent Real attention formula and full early-query FP equivalence are
connected to the same quoted bodies. Successful execution, all output cells,
exterior memory and backed active accesses remain separate proof obligations.
Local builds and source comparison do not constitute final acceptance; exact
current-source independent replay remains required in the validation records.
