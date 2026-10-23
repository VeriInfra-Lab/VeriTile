# Ordinary compiler boundary for the native plan resolver

`NativePlans.Descriptor` ranges over arbitrary source shapes. `NativePlans.Binding`
equates the selected native expression syntax with the descriptor's resolved
layout. The eight retained profile pairs are examples, not a source-domain
restriction and not evidence that every possible specialization has passed.

The producer must run `resolve.py FORWARD_RECEIPT BACKWARD_RECEIPT CURRENT_SOURCE`
on the actual specialization before using its descriptor as native compiler
evidence. An unrecognized artifact is rejected; it is not assigned a convenient
covered plan. The current recognizer handles scalar FP32 SIMT dot lowering with
four warps, two stages, SM90, and fusion enabled. It never compiles, executes a
kernel, or assumes a floating-point dot/reduction/output equality.

The explicit ordinary compiler contract is:

1. The retained receipt faithfully identifies the compiler options, actual ABI
   specialization, and generated TTIR/TTGIR/LLVM/PTX artifacts. Current kernel
   ASTs must equal the compiled ASTs. Pointer arguments denote their ordinary
   allocated typed FP32 buffers; the producer's separate bounds/frame proof
   supplies address validity, no-wrap, masks, and launch/store coverage.
2. TTIR/TTGIR SSA values and debug source sites preserve the recorded source
   operand identities, loop iteration and accumulator identities, and select
   predicates. In particular, zero seeds, prior output accumulators, dZ loads,
   and value-shard-zero selection refer to the corresponding source values.
   A shared-memory slot in a `dot_op` conversion denotes the indicated matrix
   operand element. Four local normalizer registers denote consecutive column
   elements `4 * columnOwner + register`; the retained blocked layout is
   `sizePerThread=[4,4], threadsPerWarp=[4,8], warpsPerCTA=[4,1], order=[1,0]`.
   The normalizer store selects column owner zero. These are compiler data
   routing/ABI facts, not numerical statements about a whole dot or reduction.
   For the admitted T=1 specialization, the hoisted constant DK coefficient
   supplies exactly the diagonal query columns whose source time index is
   outside T. Its direct product operand order is separately retained; the
   dynamic valid-query coefficient keeps the swapped product order.
3. LLVM scalar/vector FMA calls and PTX operations have their ordinary local
   instruction meanings. Vector extraction, insertion, constant shuffle, typed
   shared load, and pointer arithmetic use their ordinary structural meanings.
   The checker independently expands these vector operations and checks both
   operand address sequences for every scalar chain and all 128 CTA threads.
   A sequence advances through reduction indices in increasing order; the
   typed `dot_op` slot interpretation supplies its logical origin at index zero.
   LLVM `or disjoint` and identical incoming address PHIs are interpreted only
   by their stated integer semantics.
4. The local scalar instruction semantics remain the producer's explicit
   numerical obligations. No algebraic reassociation, commutation, zero law,
   final-memory premise, successful-run premise, or tensor callback is added.

Checks outside that contract include artifact hashes, matching kernel ASTs,
actual scalar/vector FMA-chain lengths and shared-load strides, conservation
of FMA instruction counts from LLVM to PTX, every retained scalar coefficient
DAG, exact mask placement/operand order, and both normalizer trees at all eight
row registers and all eight physical column owners. Tiny diagonal operand
reversal, sharded post-dot seed ordering, and no-scale specializations are
separate syntax choices.

`Binding` alone is not an artifact certificate: its constructor can establish
syntax equality for any descriptor. Native admission additionally requires
the actual artifact checker result and the ordinary compiler contract above.
No checker result is a theorem about every future compiler version or every
unseen specialization. Host reductions are intentionally a separate producer
obligation, because these native artifacts contain no Torch host lowering.

The Python checker and its parser are trusted, reviewable compiler-boundary
code. Their soundness has not been formalized in Lean. The accompanying Lean
module proves coverage/carry properties and exact layout projections for all
descriptors and all lengths; it does not pretend to prove parser correctness.
