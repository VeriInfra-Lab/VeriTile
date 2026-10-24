# Vector mixed fp32 execution and ReLU finite input words

This batch adds nine Lean modules and three public specifications across two
tasks. The campaign now has 26 supplemental contracts across 10 tasks.
**Whole-original-program acceptance remains 0/183; all 183 are pending.**
Every original stage, dtype and branch remains required. Missing structural
coverage cannot be deferred as a numerical assumption.

Vector addition adds 18 ordered integer/bool and fp32 profiles. Integer input
is normalized at its source width, rounded directly once to fp32, and added
with concrete software `FP.add cfg` in the original operand order. The `.real`
carrier exists inside explicit fp32 precision and is identity-retagged; no
native fp64 behavior or extra rounding is inferred. The complete ascending
grid reads preceding memory and equals an independent initial-memory result
with a full frame, including equal logical region names.

`vector_addition_mixed_fp32_total_correctness` exposes arbitrary signed and
unsigned policies for undefined fp32-to-integer conversions. Its logical
totalization does not prove native poison, traps or sanitizer outcomes.
`vector_addition_mixed_fp32_defined_correctness` constrains only active
integer-output conversions to their finite toward-zero range domain. On that
domain the complete stored result is independent of every undefined policy.
This is a theorem domain, not an added runtime guard. The fp32-output
orientation has no finiteness/domain premise, and masked lanes remain
unconstrained. Bool output stores a signed i8 conversion: 3 remains 3, while
128 is outside the defined domain and retains the explicit policy.

Cumulative logical value/grid coverage is 124/217 promotion-success pairs:
25 floating, 81 integer/bool and 18 mixed fp32. The remaining 93 are 54 mixed
fp16/bf16/fp64 pairs, 11 fp64 floating pairs and 28 promoted fnuz pairs rejected
by the pinned sm90 frontend. This denominator differs from the exact-source
frontend matrix, which observed 189 generated TTIRs and 100 errors across 289
pairs. This batch reuses 18 byte-identical original TTIRs and runs no native
backend compilation or GPU experiment. Original error branches remain open.

`relu_word_grid_correctness` refines nine of the previous twelve logical
profiles using explicit signed/unsigned BitVec8/16/32/64 and raw bool BitVec8.
Signed and unsigned decode have range and source-word roundtrip proofs. All
256 bool bytes normalize by nonzero to unsigned i1. Small inputs extend only
for comparison; source-width conversion supplies the selected fp32 value.
The full tagged `InputWords` relation binds every input offset below N and
does not assume execution success or output correctness. The theorem
constructs actual typed-grid success from this relation.

The ReLU reference preserves the original pid-zero-only store. Every write
has an fp32 tag, and all remaining memory—including the tail beyond the first
block—equals the arbitrary prelaunch memory. Logical x=y is permitted. A
separate y!=x constructor lemma preserves a supplied output initialization;
it does not establish native allocator freshness or physical disjointness.
Native source bytes satisfying InputWords, endianness, tags, addresses,
alignment, mixed-width views, compiler/runtime errors and instruction
interleavings remain structural gaps. fp64, fp8 errors and host/harness/module
scope are not closed by the nine finite-word profiles.

Both mixed declarations conservatively use backend-value metadata. The
defined-domain result has no unclosed numeric callback in its active output;
the total theorem retains its explicit undefined policies. No closed-bit
interpreter command is claimed. ReLU conversion and fp32 zero callbacks
remain arbitrary; inherited floating comparator parameters are unused by
the integer/bool word graph.

The independent reviews read all nine sources and checked captured exits,
dependency fingerprints and actual original TTIR store dataflow. Mixed fp32
has six successful compilations, 25 named axiom checks, four noncircularity
checks and one benign unused-simp warning. ReLU words has three successful
compilations, 23 named axiom checks, two noncircularity checks and no warnings.
The mixed revision changes only five trailing spaces in its grid source;
the original frozen package remains intact. The independent review and root
integration bind the actual lakefile.toml omitted by its producer snapshot.

[root-review.json](root-review.json) binds all new and retained task sources,
metadata and independent reviews. All preceding 23 declarations and source
views are unchanged. [numerical-followup.json](numerical-followup.json)
preserves the prior 11 rules and adds five narrow native-numeric followups.
These descriptions still require exact unified-queue scope bindings before
acceptance. Native memory, error and compiler gaps are never closed by them.
Sources are stored only in their repository locations; caches, symlinks and
drafts are excluded from this checkpoint.

Actual registered-build, sharding-test and global-gate results are recorded
in [validation.json](validation.json). Earlier 210 benchmark-tool tests are
reused only after checking unchanged tool/test inputs. Historical and exact
51518a8c comparator replays exclude all nine modules in this batch.

The actual registered Lean build passed all 7,303 jobs. Catalog generation
passed on its first attempt, and all 25 sharding tests passed. The global
audit passed eight gates and exited 1 at exactly three gates: scope review
reports 183 unaccepted original programs, while comparator preflight and
library replay both report the official comparator unavailable to the host
runner. The global runner did not run corpus replay. These observed results
validate this partial extension; they do not establish whole-program or
native-backend acceptance.
