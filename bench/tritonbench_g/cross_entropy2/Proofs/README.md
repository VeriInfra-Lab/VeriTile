# cross_entropy2

Current repaired Python implementation: [cross_entropy2.py](../cross_entropy2.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` quotes both repaired Python function bodies statement for
statement. It retains the int32 label pointer and explicit signed int64 widening,
stride casts, class-ownership tests, zero-padded load followed by scaling and the
separate negative-infinity selection, all split/smoothing branches, three forward
stores, ignored-row zero selection and the original backward multiplication order.
Caller pointer bases, dtype profiles, constexpr flags and integer geometry bindings
are outside the quotation. Runtime FP words and `total_classes` keep the existing
input contract and numerical domain.

[Python index candidates](../cross_entropy2.py) explicitly record the
existing integer-index equivalence pair. Each changes only the order of
`col_block_idx * BLOCK_SIZE`; [source correspondence inventory](../../../../validation/source-correspondence.json)
checks all four complete bodies. The repaired Python source is unchanged.

The optional condition-grouping frontend supplies parentheses required by the
internal grammar for a Python `and` chain and a bare assignment conditional.
Its independent typed reference checks the complete resulting program, including
runtime conditions, branch order, literals, implicit FP32 store conversions and
both stores. Unsupported incomplete conditionals are rejected.

The Real entries use independent forward/backward mathematical formulas. Ordered
FP expressions retain actual FP32 loads, the explicit casts and scaling order;
complete execution, safety, output and exterior-frame obligations are checked
for the actual source quotations. Validation and independent replay status are
recorded separately below.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
