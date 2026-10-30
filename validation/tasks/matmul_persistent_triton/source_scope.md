# Source and numerical scope

Pinned source: `matmul_kernel_persistent`, line 21, revision `603e28a`.
SHA256: `4537e22452bcb9df7844521498b61232772d7ae44e9aa3220c38c2b341f2152c`.

The full fp16/E4M3 source and both variants are present. Signed `tile_id =
start_pid - NUM_SMS` and `ki = -1`, flat range `k_tiles * tiles_per_SM`,
absolute tile remainder in the shortened group, all input clamping, K masks,
fused dot, output cast and accumulator resets are retained. Only integer
output-row multiplication order changes in the candidate.

The positive-K mathematical contract uses the source's full real erasure.
Independent original A/B arrays determine every result. Original input tags,
logical allocation bounds, finite disjoint allocations, complete successful
execution, actual source access safety, A/B preservation and all other raw
flat cells are included. Output injectivity is required only on logical M*N
addresses. The actual host grid covers every logical element and assigns
distinct output windows to distinct legal PIDs. This is per-program execution
plus coverage/nonalias facts, not a concurrent interleaving semantics theorem.
The same public theorem includes the standard
`ComputeCorrect.RealizesFrame_without_Rounding` region surface using exact
`MemCell` observations, the complete assigned active-output write map, the
independent GEMM expected values and the complete raw frame. This is derived
from the actual full-source execution; the successful finite-flat contract
remains a separate conjunct. K=0 has a standard empty-write-map frame and a
named successful finite-flat full-memory-preservation contract.

The public mathematical contracts use unbounded Nat/Int arithmetic. Signed
source operations have not been replaced by truncated natural subtraction.
The separate [integer-domain results](native_integer_scope.md) prove agreement
of a handwritten source-mapped signed-i32 projection under explicit sufficient
shape, stride and intermediate bounds. Reached control and coordinate values
are derived inductively; successful execution or prepared states are not
assumed. Full typed-source/TTIR refinement remains open. Mathematical positive
block sizes do not imply legal native Triton configurations, and native bounds
have not been added to `RealIO.Admissible`.

The [hint results](hint_scope.md) derive an exact criterion for the original
cached coordinate promises: when K is positive, all reached vectors satisfy
full-block contiguity and aligned starts iff BM divides M and BN divides N.
Each nonzero remainder yields an actual reached counterexample. K=0 reaches
no hint. These predicates describe the documented hint interpretation and the
source-mapped control projection; they are not compiler soundness axioms or
proofs of a concrete incorrect native result. No public theorem was narrowed.

Two concrete original-AST offline compiles are recorded in
`Proofs/native_lowering_sm90_triton_3_7_1.json`: Triton 3.7.1/sm90, BM=128,
BN=256, GM=8, NUM_SMS=132, BK=64 for fp16 and BK=128 for E4M3. Runtime
dimensions/strides in those signatures are i32. Compilation does not prove
runtime overflow safety, arbitrary profile admission or IEEE/GPU numerics.

Typed structural FP equivalence quantifies every backend algebra and every
initial typed state, including failure results, with empty assumptions/domain.
It is not a claim that the exact-real GEMM formula equals native rounded GEMM.

The explicit fp16 macro selection now carries a pointer-dtype profile bound
to the original source and recorded native signature/TTIR. Both original arms
are validated before the known storage equality is selected. Both recorded
storage views pass the scoped production comparisons without an exclusion.
The full audit after integration remains a separate gate.
Official independent comparator replay remains pending. These checked formal
results therefore do not mark the original Triton port complete.
