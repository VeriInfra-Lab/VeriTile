# cross_entropy1

Current repaired Python implementation: [cross_entropy1.py](../cross_entropy1.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` quotes both original Python function bodies statement for statement,
including signed stride casts, masked loads, the explicit FP32 cast, split and
smoothing branches, ignored-label selection, and the original multiplication order.
Caller pointer bases, dtype profiles, constexpr flags and integer geometry bindings
sit outside the quotation. Runtime smoothing and square-scale values remain free
FP32 inputs; `total_classes` remains a runtime integer constrained by the input
contract. No additional numerical domain restriction is imposed.

The existing integer-index commutation pair now has an explicit derived Python
source in [cross_entropy1.py](../cross_entropy1.py).
It changes only `col_block_idx * BLOCK_SIZE` to `BLOCK_SIZE * col_block_idx`;
[source correspondence inventory](../../../../validation/source-correspondence.json) checks all four complete bodies.
It is an equivalence candidate, not another repaired implementation.

`forward_unsplit_kernel`, `forward_split_kernel` and `backward_kernel` prove the
original selected row/block programs against independent mathematical formulas.
The ordered FP reference retains both the FP32 load conversion and the subsequent
explicit cast. The equivalence proofs cover the complete forward/backward source,
actual scalar bindings, memory safety and exterior frames, with the existing
public domains and floating-point assumptions.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
