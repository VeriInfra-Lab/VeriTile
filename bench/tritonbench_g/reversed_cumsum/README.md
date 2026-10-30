# reversed_cumsum

The pinned source is `chunk_global_reversed_cumsum_vector_kernel` in
[`reversed_cumsum.py`](reversed_cumsum.py), from TritonBench commit
`603e28a`. Its complete typed transcription and candidate are in
[`Proofs/Source.lean`](Proofs/Source.lean). The candidate only reverses the
operands of the discrete comparison that creates the upper triangular matrix.
Both keep the complete reverse loop, loads, casts, dot, store and carry update.

The complete real execution, independent mathematical output, tagged memory
frame, finite allocation IO and typed success chain have passed local Lean
checks. The public entries are now [`Kernels.lean`](Kernels.lean),
[`Correct.lean`](Correct.lean) and [`FPEquiv.lean`](FPEquiv.lean). Their composition
passed all eight public axiom gates, with an empty FP assumption list.
Historical prepared-input slices are retained only in
[`Proofs/LegacyReal.lean`](Proofs/LegacyReal.lean). Its single-block closed forms
now state their previously implicit zero input-oracle premise explicitly; the
source loads themselves retain unspecified padding.

The source omits `padding_option`. Its general real output is therefore the
initial input suffix **plus the sum of unspecified values in the final time
tile**. [`Proofs/RealExecution.lean`](Proofs/RealExecution.lean) proves this from
the initial state, for every time dimension including zero. It requires a
positive time block size, distinct input/output regions and injective active
output addresses. [`Proofs/RealFormula.lean`](Proofs/RealFormula.lean) connects
that result to the conventional suffix formula. The finite memory theorem in
[`Proofs/RealIO.lean`](Proofs/RealIO.lean) explicitly requires the zero oracle
used by the current flattening bridge, together with allocation bounds and
disjointness; it preserves every untouched memory tag and relocated pointer.

[`Proofs/FP.lean`](Proofs/FP.lean) proves equality of complete typed executions
for arbitrary arithmetic algebras, initial states, shapes and independent
input/output storage types. The separate typed success construction uses a
`WithBot ℝ` consistency algebra with actual mathematical dot and reduction
operations, for independent bf16 input and fp16 output at every layout and
initial state. A concrete witness starts with empty registers, nonzero program
IDs and both partial tails at T = 17, S = 33, BT = 16 and BS = 32. It does not
assert IEEE rounding or GPU numerical correctness.

The source AST was mechanically compared in full with the pinned Python,
including negative controls. Offline native compilation checked all nine
autotune choices and additional independent storage combinations, including
empty and partial time tiles. These checks establish source agreement and
compiler admission, not a formal native refinement. See
[`source_scope.md`](source_scope.md) for the remaining ABI and host boundaries.
