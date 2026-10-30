# kldiv_ops checked repair

The original benchmark is unchanged. `kldiv_ops_repaired.py` is the checked,
contiguous FP32 CUDA candidate. The previous broad candidate and all its valid
proofs remain under `kldiv_ops_conditional.py`, `README.conditional.md`, and the
`ConditionalCorrect`/`ConditionalFPEquiv` imports.

## Checked domain

Forward admits matching rank-two, strided, contiguous FP32 CUDA tensors on one
device. Backward applies that same contract to target and the supplied
destination; its cotangent is scalar or has exactly the matrix shape. All
arguments must have resolved negative flags and aligned active FP32 pointers.
The forward read inputs may alias. A nonempty backward destination must have
backing storage disjoint from both target and cotangent. This conservative
storage check rejects separate slices of one allocation too; empty destinations
skip it. Nonzero offsets, unused degenerate strides and both empty axes remain
admitted. There is no size cap or tensor-value guard.

`log_target` is a Python bool, reduction is one of `none`, `sum`, `mean`, and
`batchmean`, and epsilon is a Python float. Every epsilon value and FP32 tensor
word is admitted, including NaNs, infinities, signed zeros and subnormals.
`Contract.lean` and `Proofs/Checked/Contract.lean` prove equivalence between those
ordered checks and the public domain, including non-Tensor/type classification.

The candidate passes the original input objects and supplied destination to the
kernels directly. It does not resolve, pack, clone, allocate a backward work
buffer, or copy back. The owner device context covers launch and postprocessing.
Forward retains the original `(BT,)` allocation for reduced output, the same
`sum(dim=0)` versus `sum()` calls and their actual denominators. Backward writes
the unscaled prediction derivative, returning the exact destination for a scalar
one cotangent and the actual Torch product otherwise. The caller supplies any
forward-reduction scaling through the cotangent.

## Source and mathematical result

The historical repaired JIT AST is unchanged: positive block size, i64 indices,
and explicit remasking of inactive loss lanes only on the reduced branch. Each
row carries an FP32 accumulator across chunks. This is not a chunk-buffer
redesign. The formulas are `t*(log(max(t,eps))-y)` and `exp(t)*(t-y)`; there is no
special extension of the raw-target formula at zero.

Public `Contiguous.Correct.forward` and `.backward` prove the independent Real
interpretation for the actual whole calls. The forward result is elementwise
loss, total loss, total divided by `BT*V`, or total divided by `BT`; backward is
`-t` or `-exp(t)` followed by the entry cotangent. Lean's Real division and log
are totalized. This ideal interpretation is separate from native exceptional
word behavior and makes no FP32 accuracy claim.

The preserved source proofs establish complete typed rows and grids, masks,
coverage and sequential chunk accumulation. `Checked.Source` binds them to the
original checked physical descriptors. `Checked.Direct` proves that full row
source executions against evolving physical memory equal the public store graph.
The source loads remain stable through every store prefix by fresh forward
allocation or the checked backward separation. `Checked.Execution` proves exact
active native addresses, all four backed bytes, alignment, signed loop masks,
terminal increments and PID representation. Bounds come from actual backing;
successful dispatch supplies only ordinary PID representability.

## Native values, bytes and remaining scalar obligations

Public `Contiguous.FPEquiv.forward` and `.backward` prove the actual returned
objects, raw FP32 words and all four bytes. The independent native graph retains
source operand roles, masked zeros, explicit compiler-selected addition trees,
sequential chunk carry and the actual returned Torch reduction plan. Epsilon is
represented by its Python binary64 bits with an explicit scalar ABI conversion
to the kernel's FP32 epsilon. Literal multiplication by minus one and unary
negation have separate scalar ports.

Optional `forward_reference` and `backward_reference` transport this result
using only the fixed-arity scalar equalities listed in
`numerical-obligations.json`: kernel add/sub/mul/max/log/exp/negation/literal-minus-
one/epsilon conversion, plus Torch add/mul/equality/integer-scalar division.
There is no reduction, output, address, frame, execution-success, or memory
predicate registered as a numerical port. Addition trees and their coverage are
structural data, not whole-sum equalities. Installed compiler and runtime
selection of those local primitives/trees remains an explicit ordinary backend
boundary. No GPU conformance or universal correctly rounded transcendental
behavior is asserted.

The host proofs cover ordinary allocation, context, launch setup, reduction,
comparison, product and division outcomes. Forward preserves every byte of
arbitrary initially live backing allocations. Backward preserves the complete
destination complement, including padding and partial-byte tails. Writes are
retained if later comparison or multiplication fails. Successful calls restore
the previous device. The Real memory carrier has no impossible encoding into
finite bytes; only native words use the checked four-byte codec.

## Validation

`check_contract.py` passed 38 rejection cases, 5 ordinary failure prefixes,
182 CPU Triton source/reference/frame cases, and 3 original-source
counterexamples (inactive reduction NaN with epsilon zero, logical-stride
mismatch, and unresolved negative backward target). Cases cover both flags,
all modes, both empty axes, positive/zero/negative/subnormal/infinite/NaN
epsilon, tail/chunk boundaries, offsets, alias rejection, destination identity,
products, and allocation frames. These are CPU interpreter controls; no GPU
was queried. Public and supporting Lean modules pass standard-base axiom audits.

Source hashes are recorded and checked in `source_pin.json`:

- Original: `36512c93f35738488b343c4b2db4a2bb5b11ae6967976673099794efa9d8c2b1`.
- Checked candidate: `5a610a722d441f13cccdf9ef8443f994046b26bd1fbec5f858c12d1f8809a20d`.
- Historical conditional candidate: `90fa8cf3819ccbbdb72f10c20e050fa4faed6d01bad2ead31a46b3d4208fc446`.
