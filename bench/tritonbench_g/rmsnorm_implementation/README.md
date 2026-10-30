# rmsnorm_implementation

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

This port proves both complete two-pass RMSNorm programs in its floating,
nonnegative-stride model. The five public contracts comprise two preserved
historical real headlines, the earlier typed real contract, a stronger
`full_real_correct` contract, and full structural FP equivalence. The named
`FullExactIO` contract combines actual trace safety, successful finite flat
execution, the independent full-row output formula, and a frame for every other
raw cell. The entire source domain and independent comparator replay remain
unfinished; see `source_scope.md` for explicit boundaries.

The source retains both loops, both program IDs, all seven strides and all
memory operations. Its 216 profiles independently choose input, weight and
output floating storage. Pass one casts input to fp32 and accumulates squares
there; sum, division and sqrt stay fp32. Pass two reloads the original input,
divides by the fp32 standard deviation and multiplies the uncast weight. Either
operation may promote to fp64, and the final store converts to its destination
storage. Input loads specify zero padding; the masked weight load retains its
missing `other` argument. `Proofs/Precision.lean` checks the exact load,
arithmetic and store prescriptions for every profile and variant.

The independent real formula is

`x[t,j] / sqrt(sum_{u,e : u*B+e<N} x[u,e]^2 / N + eps) * w[t,j]`.

The dimension-general real contract makes finite allocation, access bounds,
positive block/output strides, output/input separation and the exact model's
zero-undefined convention explicit. Positive epsilon implies a positive
denominator; the general formula retains Lean's total division convention.

The candidate commutes one natural address multiplication. The FP theorem
compares complete executions for an arbitrary backend, including failures,
aliases, undefined masked values, all registers and all memory. It uses no
numerical rewrite assumption.

`Proofs/FiniteWitness.lean` discharges every real IO premise for all 216 profiles
and both variants: two nonzero program IDs, N=3/B=2, nonunit strides, finite
X/W/O allocations of 18/10/32 cells, three outputs equal to 2, integer-37
sentinels in a stride gap and masked tails, and the complete raw frame.
`Proofs/Execution.lean` separately executes the genuine typed programs with
actual sum/square-root arithmetic and identity casts. Four complete runs cover
both variants and both fp64-promotion locations; all produce standard deviation
3 and three bf16 output cells equal to 2 while preserving every other typed cell.
These are exact-real backend witnesses, not IEEE or GPU tests.

The native compiler receipt checks the unchanged AST on Triton 3.7.1 / sm90:
32/36 floating wrapper profiles and four additional floating destinations
compile with expected precision; four fp8/fp64 wrapper profiles fail on
unsupported conversions. Four admitted integer/Boolean probes expose remaining
unmodeled source paths. No native dtype guard is invented.

`Proofs/Audit.lean` checks public and support axioms, independent specifications,
FP assumptions and every theorem in the port. All 111 historical declaration
headers and 50 adjacent docstrings from the pre-split source remain unchanged.
The public entries reach 13 local modules; the separate audit reaches all 14.
`completion_audit.md` records the final build and review receipts.
