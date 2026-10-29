# streamk_matmul — full pinned source SC contracts

The unchanged upstream source is pinned at revision `603e28a`, SHA256
`794c7aa68254a0cc1caf974b5ed806ce8c884e9f7c9c2c36d68c67d5b654a4f3`.
The previous task had no public Lean declaration; its README accurately marked
both source import and proof missing. No earlier accepted theorem is replaced.

`SourceCorrect.lean` provides parameterized source execution and value contracts
for both complete kernels and the two ordered launches. All five JIT functions
are translated, including both tile mappings, every dynamic while, the full K
range, CAS return-value polling, publication followed by release, and per-lane
atomic additions. All 64 original JIT statements are ledgered; inlined helper
copies produce 75 ledger entries. No loop is unrolled in a public theorem.

The execution model is an explicit sequentially consistent lane serialization.
An arbitrary finite scheduler chooses programs at each step; loads, stores and
RMWs are individually interleavable, with row-major lane order inside each
instruction. Failed CAS attempts remain real steps. Signed i32 division faults
remain faults; control and pointer offset arithmetic preserve 32-bit wrap and
separate sign-extension at pointer additions. This is not an unbounded-integer
substitute for native scalar arithmetic. Native i64 specializations are not
claimed. The Python parser, shape checking and helper inlining are an explicit
unverified frontend boundary; Lean checks and executes the resulting entire IR.
The independent original-Python CPU replay and compiled original AST artifacts
supply additional checks, not a general verified Python frontend theorem.

The reference contains only symbolic expressions over initial memory. It has
no numeric accumulator, source execution call or program counter. An independent
integer-only control runner determines event addresses and ordering. Each load
substitutes the current memory expression; dot, separate accumulator add,
output casts, half conversion and atomic addition remain ordered expression
nodes. The theorems prove every data cell, every lock cell and every register
against this reference, plus explicit data and lock frames. Logical data
aliases are supported. No ready-lock, zero-input, finite-value, known-output,
step-correctness or no-concurrency premise is introduced.

The host plan retains `two_tiles`, full/partial iteration partitioning,
contiguous strides, `GROUP_M=8`, stream/full grids and actual zero initialization
of the allocated lock cells. These host formulas have exact AST checks and an
explicit manual-lowering boundary; no general torch host/allocator translation
theorem is claimed. Uninitialized C is unconstrained. A completed host
result is returned only when both computed schedules finish without a fault;
otherwise the result is pending. Separate per-wave contracts still describe
all blocked and incomplete prefixes. There is no theorem that every schedule
terminates, that a GPU occupancy schedule is fair, or that the protocol is
race-free under a GPU memory model.

The generic backend contracts retain all original arithmetic roles. The concrete
fp32-storage profile closes RNE fp32 add, separate multiply/add software dot,
fp32→binary16→fp32 rounding and fp32 atomic-add value steps for every bit pattern.
The public fp32 callback profile also permits any explicitly chosen dot primitive.
The executable sequential-dot specialization is a software schedule, not a
native default-dot claim. fp16/bf16/integer storage still use abstract backend
bindings; they are not silently counted as concrete bit models.

Original behavior and defects remain visible:

* No A/B/C operation has a tail mask. A kernel-checked witness uses the native-
  compilable block shape 32 with host M=N=K=1 and reads A[1] outside allocation.
  The theorem uses logical memory and does not certify allocation safety.
* Full tiles unconditionally convert to fp16 before storing even into fp32 C.
  A complete execution loses the fp32 low bit of `0x3f800001`.
* Three-program, zero-lock, nonzero executions produce fp32 1 or 0 solely by
  changing the atomic contribution order. The result is intentionally schedule
  dependent; it is not replaced with an associative real sum.
* A scheduler omitting the publishing program leaves the original CAS loop
  blocked. The model does not force readiness or exit after a bounded retry.
* Native SM90 offline compilation preserves all five original JIT ASTs. Both
  kernels compile for fp32/fp16/bf16, and int8 full tiles compile; int8 first wave
  fails because `atomic_add` does not support int8. No GPU was executed.
* Native fp32 TTIR uses `inputPrecision=tf32` and fuses source dot-plus-add into
  a dot accumulator. The source model keeps the separate add. Neither TF32 nor
  fusion equality is assumed; a compiled-graph value proof remains separate.

`Witness.lean` contains kernel-checked completed three-program order examples,
a five-output two-wave execution with multiple outer chunks and K iterations,
blocked polling, half-rounding loss, and the allocation counterexample. These
support the parameterized contracts; none replaces their full scope.
