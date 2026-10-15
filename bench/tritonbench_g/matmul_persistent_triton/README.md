# matmul_persistent_triton

Status: IN PROGRESS. Complete exact-real source contracts and typed structural
FP equivalence are checked; native/source admission, full integrated audit
and official independent replay remain open. This is not complete
original-Triton-port acceptance.

The pinned Python source is revision `603e28a`, SHA256
`4537e22452bcb9df7844521498b61232772d7ae44e9aa3220c38c2b341f2152c`.
Both fp16 and E4M3 wrapper profiles retain signed negative initialization,
the exact single `k_tiles * tiles_per_SM` loop, the absolute-tile grouped
remainder, clamped M/N input lanes, K masks, fused dot, casts and resets.
The candidate only commutes the signed integer output-row product.

`real_correct` retains `RealIO.Implements` and adds the standard
`ComputeCorrect.RealizesFrame_without_Rounding` region surface for
both profiles and variants after dtype erasure. From original tagged A/B
memory, positive dimensions/block sizes, the actual host grid
`min(NUM_SMS, ceil(M/BM)*ceil(N/BN))`, valid PID, finite allocation bounds,
output/input region separation and logical output-stride injectivity, it
proves successful full-source flat execution, the independent input-only
GEMM in every assigned active output, input preservation and the complete raw
memory frame. The shared flattening bridge's `undef = 0` convention is explicit.
There is no prepared-register, prepared-accumulator or assumed body-execution
premise. `RealIOCoverage` covers every logical output and proves disjoint
active windows for distinct legal PIDs.

The standard region surface observes exact `MemCell` values through a masked
write map covering every assigned persistent output tile. Its expected values
come from the independent input-only GEMM formula; its frame preserves all
cells outside that same write map. `RealSurface.outside_iff` proves equality
of this complement with the original full-source raw frame. The stronger
finite-flat contract remains a conjunct, so termination and allocation safety
are retained alongside the standard partial-correctness surface.

`zero_K_preserves_memory` separately proves the actual K=0 behavior through
the standard empty-write-map region frame and a named universal successful
finite-flat preservation contract. All memory remains unchanged, including
nonzero output sentinels; the source does not write the zero matrix.

`FPEquiv.source_equivalence` proves complete typed-state/failure equivalence
under every backend algebra, with empty FP assumptions and empty domain.
It leaves every floating operation unchanged. This is separate from exact-real
GEMM correctness and does not establish native IEEE numerical results.

The preceding helper checkpoint contains 76 Lean modules in 79 files. Its
critical symbolic checks total 116 standard-axiom outputs, plus the original
12 baseline checks. The integrated port now has 96 Lean modules, including
the public entries, ten native-integer helpers, five hint helpers and their
two separate axiom-audit modules.
The public entries are independently compiled and audited in the candidate
integration bundle. `RealFiniteWitness.values` supplies a fully discharged
finite allocation with M/N/K tails, two persistent outputs, stride gaps and
18/27/36 readbacks while retaining the entire raw-cell frame. Its small formal
block configuration is not claimed to be a legal native GPU configuration.

Proof-side dimensions, strides and integer intermediates use unbounded Nat/Int.
`RealIO.Admissible` gives mathematical shape/region/stride conditions only;
it does not impose native i32 bounds or legal Triton configurations.
The separate `NativeDomain.Admissible` gives explicit sufficient arithmetic
bounds for a source-mapped BitVec control/coordinate/address projection. Ten
checked helpers establish its reached-state agreement, including negative
initialization, inactive-tail arithmetic and the two separate C pointer
advances. All 76 axiom checks pass; three require no axioms.

Five further helpers characterize the source's cached-coordinate hints.
For positive K, every reached vector has the promised full-block contiguity
and aligned start exactly when BM divides M and BN divides N. Any nonzero
M/N remainder constructs a legal PID and actual hint-refresh iteration with
a noncontiguous vector. For example, M=129 and N=257 produce cached vectors
starting `[128, 0, ...]` and `[256, 0, ...]` in the recorded configurations.
K=0 reaches no hint. Independent local Lean replay reproduced all five
artifacts and 43 axiom outputs, with no diagnostics.

These are handwritten source-mapped projection results. They do not establish
full typed-source/TTIR refinement, compiler soundness, a concrete bad GPU
result, arbitrary native admission, concurrent launch behavior or IEEE
numerics. The existing public mathematical and FP statements are unchanged.

The original AST has two concrete offline Triton 3.7.1/sm90 compile receipts:
BM=128, BN=256, GM=8, NUM_SMS=132, BK=64 for fp16 and BK=128 for E4M3.
They prove neither arbitrary native-configuration admission nor runtime
fixed-width safety, and include no GPU execution.

The selected fp16 textual view records an explicit pointer-dtype profile.
The audit checks the original source hash, exact Lean selector/arguments, all
pointer storage declarations and the original native signature/TTIR before
selecting the dtype equality branch. It validates every original arm first;
unrelated runtime guards remain. Both recorded storage views pass their
scoped production source gates without a translation exemption. Full audit
and public build acceptance remain separate from these profile checks.

See `source_scope.md`, `native_integer_scope.md`, `hint_scope.md` and
`completion_audit.md` for the exact acceptance scope.
