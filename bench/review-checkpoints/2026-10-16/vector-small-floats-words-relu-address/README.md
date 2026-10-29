# Small floating profiles, integer words and ReLU finite addresses

This batch adds twelve Lean modules and four public specifications across
two tasks. The campaign now has 30 supplemental contracts across 10 tasks.
**Whole-original-program acceptance remains 0/183; all 183 are pending.**
All original declared stages, dtypes and branches remain required. Only
explicitly registered numerical correspondence may be deferred; structural
omissions remain blocking.

Vector addition adds 36 ordered integer/bool and fp16/bf16 profiles. Half
profiles normalize the integer at its source width, round it directly to
binary16, and use binary16 addition. Bfloat profiles widen the bfloat operand
to binary32, round the integer directly to binary32, and add in binary32.
Half sums widen to binary32 before integer output conversion; bfloat outputs
narrow with the concrete software cast. Original operand order is preserved.

The software carrier stores half/bfloat payloads in its low 16 bits and
binary32 payloads in all 32 bits. This carrier choice does not interpret
half input bits as binary32 numbers. The explicit `.real` cast is an identity
carrier retag after profile-specific conversion, not native fp64 or an extra
rounding step. The total theorem exposes arbitrary policies for undefined
integer conversions. The defined-domain theorem constrains only active
integer outputs and proves policy-independent memory. Floating outputs need
no finiteness or integer-conversion premise, and masked lanes are unconstrained.
Bool output remains a signed i8 conversion, including raw byte 3.

Cumulative logical value/grid coverage is **160/217 promotion-success pairs**:
25 floating, 81 integer/bool, 18 mixed fp32 and 36 mixed fp16/bf16. The remaining
57 are 18 mixed fp64, 11 fp64 floating and 28 promoted fnuz pairs rejected by
the pinned sm90 frontend. The exact-source frontend matrix observed 189 TTIRs
and 100 errors across 289 pairs. This batch reuses immutable captured TTIRs;
it performs no new frontend compilation, native backend compilation or GPU
execution. Native compiler/error/poison/sanitizer behavior remains open.

The integer-word theorem refines the existing 81 integer/bool pairs, adding
no dtype pairs. Explicit source-width BitVec inputs bind complete tagged
logical cells. All 256 bool input bytes normalize by nonzero, while output
bytes are preserved: bool-plus-bool wraps at compute i1, and a bool output
can retain raw byte 3. Independent word arithmetic gives complete ascending
grid success, exact output readback and a full initial-memory frame. Equal
logical input names require the two decoded cells to agree; the construction
lemma states this compatibility explicitly. No floating callback is used.

The ReLU address theorem refines the previous nine word profiles. With the
original fixed block size 1024 and `N < 2^31`, every launched lane, including
masked tail lanes, has exact signed-i32 multiply/add, mask and pid-zero gate.
A separate lemma executes the actual original typed source's first four
statements and relates its offsets/mask registers to the finite expressions.
Explicit 64-bit mathematical base/end bounds establish active byte-pointer
arithmetic and complete element footprints, including each last byte.

The public ReLU theorem combines existing word-grid success, exact memory
and frame with that arithmetic certificate. Only pid zero stores; arbitrary
prelaunch memory beyond the first block is preserved. The base intervals are
not connected to native allocation or byte loads by a memory relation.
Native alignment, provenance, layout, bool pointer lowering, mixed-width
aliasing and instruction interleaving remain open. `N = 0` proves only the
logical empty-grid result; native zero-grid behavior, `N >= 2^31` and other
block sizes are outside this address subtheorem.

Independent reviews read all twelve sources and checked actual compilation
logs, fingerprints, public statements and scope boundaries. Mixed small
floating profiles passed six compilations, 32 named axiom checks and four
noncircularity checks; integer words passed three compilations, 17 axiom and
two noncircularity checks; ReLU addresses passed three compilations, 21 axiom
and three noncircularity checks. All twelve module audits passed without
warnings. The mixed proof's earlier stack overflow is retained separately
with its exact failed source/log and repair diff; it is not counted as a
successful run. Local proof transparency controls avoid expanding scalar
rounding algorithms without changing their definitions or conclusions.

[root-review.json](root-review.json) binds reviewed sources and independent
reviews. All prior 26 declaration records and source views are preserved.
[numerical-followup.json](numerical-followup.json) preserves all 16 prior
rules and adds four narrow mixed-format numerical followups. The inherited
ReLU conversion/zero obligations are not duplicated. These descriptions still
require exact unified-queue scope bindings before acceptance; source, memory,
address and native error gaps cannot be deferred through them.

[validation.json](validation.json) records observed registered-build,
catalog, sharding-test and global-gate results. Earlier 210 benchmark-tool
tests are reused only after checking their unchanged tool/test inputs.
Historical and exact-51518a8c comparator replays exclude all twelve new
modules. Sources live only at repository source paths; temporary caches and
symlinks are excluded from retained checkpoint artifacts.

The actual registered Lean build passed all 7,315 jobs. Catalog generation
passed on its first attempt, and all 25 sharding tests passed. The global
audit passed eight gates and exited 1 at exactly three: scope review reports
183 unaccepted original programs, while comparator preflight and library
replay report the official comparator unavailable to the host runner. The
global runner did not run corpus replay. These observed partial-extension
results do not establish whole-program or native-backend acceptance.
