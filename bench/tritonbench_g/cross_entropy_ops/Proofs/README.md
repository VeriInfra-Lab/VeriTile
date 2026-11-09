# cross_entropy_ops

Current repaired Python implementation: [cross_entropy_ops.py](../cross_entropy_ops.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The checked entry above imports the resident JIT functions unchanged from
[cross_entropy_ops.py](../cross_entropy_ops.py). The source
inventory explicitly selects that file and checks every task-local JIT, including
the [existing integer-index candidates](../cross_entropy_ops.py).

`Kernels.lean` quotes both resident function bodies statement for statement.
It retains every signed cast, masked load, FP32 conversion, logit scaling,
smoothing/split branch and the original multiplication order. In particular,
`z_loss_ptr` is stored only when `not SPLIT`, and backward retains the unused
`smooth_positive` assignment and the explicit ignored-row zero selection.
Caller pointer bases, dtype profiles, flags and integer geometry bindings are
outside the quotation. Runtime FP words and `total_classes` keep their existing
input contract and numerical domain.

The candidate source is explicitly derived for the existing equivalence pair:
only `col_block_idx * BLOCK_SIZE` is commuted. All four complete bodies are checked
by [source correspondence inventory](../../../../validation/source-correspondence.json). The Real entries use
independent mathematical formulas; ordered FP expressions retain actual FP32
loads, explicit casts and scaling. Complete source execution, access safety,
outputs and exterior frames include the untouched z-loss region in split mode.
Local validation and independent replay are recorded separately below.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
