# kldiv_compute checked direct repair

The checked entry uses the original JIT unchanged and computes the pointwise
term `x * log(x / y)` directly into its first `empty_like(x)` allocation. It
checks inputs before allocation, launches once for nonempty input under the
owner's CUDA context, and returns that same Tensor. There is no input packing,
temporary output, copyback, dtype conversion or arithmetic rewrite.

The unchanged original reads flat physical offsets. For `x=[1,1]` and
`y=[1,1,2][::2]`, it returns `[0,0]` while the logical pointwise result is
`[0,-log(2)]`. The checked entry explicitly rejects this unsupported strided
input. The separate historical generalized repair and its applicable proofs
are preserved: `kldiv_compute_conditional.py`, the conditional public entries,
and `revision002/` retain the old packing/copy models and their precise premises.
They are distinct from the current direct checked API.

## Exact checked contract

Both arguments are strided CUDA FP32 Tensors with equal shape/device,
contiguous storage and no unresolved negative-view bits. Nonzero offsets,
shared/overlapping read allocations, arbitrary rank, scalar shapes and empty
dimensions are admitted. Active data pointers have four-byte alignment.
`numel < 2^31` ensures the original signed32 offsets, mask and 1024-lane launch
remain exact, including the final inactive tail. It also bounds the one-axis
grid to `2^21`, within the ordinary launcher limit. Empty tensors return the
fresh allocation without a launch or pointer-alignment requirement.

`Contract.lean` reuses the exact ordered `add_example` metadata predicate;
`python_checks_iff_preconditions` proves successful Python-style admission iff
that public domain holds. There is no tensor-value positivity or finiteness
guard. Negative stored values, signed zeros, infinities, NaNs and subnormals
remain in the unconditional FP32 source/byte result.

## Public proofs

The new public namespace is `KldivCompute.Contiguous`. `Correct.checked_real`
constructs the complete checked logical-grid execution and proves every output
equals the independent `x_i * Real.log(x_i/y_i)`, with all other logical memory
unchanged. Real.log and division use Lean's total Real conventions. Native
exceptional values retain the actual FP32 source graph; no native Real-accuracy
claim is made.

`FPEquiv.checked_native` constructs the finite source execution and proves the
actual returned Tensor's logical FP32 result from entry-state input values,
full exterior caller bytes, actual data-pointer binding, same shape/device/dtype
and the exact first fresh allocation identity. `Checked.Source` proves exact
AST lowering; `FPProof.typed_ast_lowering` connects it to the selected typed FP32
source. The reused historical `Program` proves the complete typed programs and
evolving grid independently. Active pointer bounds, natural alignment, signed
mask/offset arithmetic, complete coverage and all memory frames are proved.
Ordinary successful allocation, Tensor metadata, local memory read/write laws
and launch-device semantics remain explicit environment boundaries.

`FPEquiv.checked_reference_if_local` adds only three fixed-arity scalar ports:
FP32 division, the actual `tl.log` intrinsic and FP32 multiplication, in that
source order. Their domains apply at the actual input/quotient/logarithm words;
they do not change Python admission. No complete KL result, source success,
address, frame or reduction equality is a deferred numerical assumption.
`numerical-obligations.json` records the declarations, exact domains and evidence
needed for the shared numerical experiment batch. GPU conformance and intrinsic
rounding/FTZ/NaN behavior are unmeasured.

## Checks and preserved evidence

All 28 task Lean modules, including preserved conditional entries, compile with
standard-base axiom audits. The task-local native compatibility helper retains
the historical proof without changing shared APIs or the current vector-add
repair. Original Python matches the historical checkout byte for byte.

`python bench/tritonbench_g/kldiv_compute/check_contract.py` checks source
pins, exact original JIT AST equality, 15 rejection cases and 19 actual Triton
CPU-interpreter/reference/frame cases. These cover tails around 1024, empty and
scalar shapes, degenerate contiguous strides, offsets, input aliases, complete
input backing bytes with partial tails, exceptional FP32 values, direct launch
arguments, one allocation and exact return identity. It forbids packing calls
and separately reproduces the original stride counterexample. CUDA metadata and
context are adapted for CPU; these checks do not measure GPU numerics.
