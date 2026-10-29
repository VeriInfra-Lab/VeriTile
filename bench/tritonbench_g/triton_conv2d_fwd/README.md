# triton_conv2d_fwd

The original wrapper defaults to `fp16=False` and `tf32=False`. Its examples
use fp32 input and weight tensors, and its output allocation is always fp32.
The `fp16` flag controls an intermediate operand cast before `tl.dot`; it does
not mean that input, weight, or output storage is fp16.

[BranchesCorrect.lean](BranchesCorrect.lean) now covers both `fp16` choices and
both `tf32` choices in the fp32 storage profile. The default branch keeps its
fp32 operands; the true branch performs the original binary16 conversion. Both
contracts cover the entire kernel-height, kernel-width, and channel-block loop
nest, arbitrary Nat model shape/stride/padding/group parameters, zero-trip cases,
masked-load undef values, every active output, and the complete memory frame.
The original output-address injectivity condition is retained. No finite-value,
zero-padding, ideal-rounding, or supplied-output premise is added.

The independent reference reads initial memory and retains the ordered dot and
outer accumulator additions. The software specialization closes every used
callback using binary32 arithmetic and an ascending dot order; the true branch
additionally narrows to binary16 and widens its dot operands. A checked complete
2×2×2-loop execution in [FPDefaultWitness.lean](Proofs/FPDefaultWitness.lean)
produces `8.00390625` with the default branch and `8` with the half-cast branch,
so their distinction is observable in an actual stored output.

The earlier [FPCorrect.lean](FPCorrect.lean) and ideal-real declarations remain
unchanged. Their existing receipts do not cover this new default-branch proof.
The historical registry wording “fp16 input/weights/output” is incorrect for
those contracts: their actual storage is fp32, with fp16 intermediate operands.
The new default source specializes away exactly the two assignments guarded by
`if fp16`, preserving the rest of the existing source body and using a typed
`fp32 × fp32 → fp32` dot. This task-local source lowering remains an external
translation boundary.

This does not yet prove the complete Python host's three-dimensional grid and
allocation behavior, other input storage dtypes, native integer/address
specialization, or physical allocation safety. Native TF32/tensor-core order,
dot/accumulator fusion, and masked-load poison/undef behavior need explicit
binding to the software graph. The source and generic backend contracts retain
those behaviors rather than replacing them with an associative real sum.
In particular, the logical address model converts signed height/width offset
terms to Nat separately. Its masked-lane addresses and undef values still need
correspondence with the original signed pointer arithmetic; this is a structural
source-binding gap, not a numerical assumption that can be deferred.
