# Vector program permutations and ReLU mathematical byte memory

This batch adds seven Lean modules and six public specifications across two
tasks. The campaign now has 36 supplemental execution contracts across ten
tasks. **Whole-original-program acceptance remains 0/183; all 183 are pending.**
All original declared stages, dtypes and branches remain required. Only
explicitly registered numerical correspondence may be deferred; structural
omissions remain blocking.

The vector schedule results refine 135 existing ordered profiles: 81
integer/bool pairs, 18 mixed fp32 pairs and 36 mixed fp16/bf16 pairs. The five
public contracts consist of one integer result and total/defined variants for
each mixed family. They add zero dtype pairs. Combined with the earlier 25
floating permutation profiles, all 160 currently covered logical value/grid
pairs now have arbitrary complete whole-program permutation results.

Each program executes the existing typed source and backend against the memory
left by its predecessor. Local registers are reset using the same initial
context as the existing ascending grids. Same-offset locality and disjoint
program ownership give the independent initial-memory result and frame.
Equal logical region names are allowed. A complete permutation of the empty
grid gives the logical no-op result. The generic internal step premise is
discharged by the concrete source execution theorems; public contracts assume
neither execution success nor their desired result.

Programs remain atomic at this scheduling layer. Native instruction
interleavings, races, physical byte aliases, finite addresses, allocation,
source/compiler errors and host/harness/module execution remain open. The
mixed total contracts keep explicit undefined-conversion policies; the defined
contracts retain the existing active integer-output domains and policy
independence. These domains are theorem premises, not new runtime guards.
Logical value/grid coverage remains 160/217 promotion-success pairs; the same
57 pairs remain outside it.

The ReLU byte result refines the same nine integer/bool profiles. Its memory is
a total `Nat -> BitVec 8` byte map. Little-endian loads concatenate individual
bytes, split/store operations extract real eight-bit slices, and generic
roundtrips preserve arbitrary words and byte maps. Sequential four-byte output
stores equal an independently stated byte reference with a complete frame.

A complete tagged logical view connects input cells and fp32 output words to
those byte loads. Under distinct logical names and explicit full input/output
interval separation, actual existing typed-grid execution and its word result
correspond to the independently defined byte grid and byte specification.
Only the first `min(N, 1024)` output words change, matching pid-zero-only
stores; the arbitrary prelaunch output tail remains unchanged. The
construction lemma supplies a logical state for arbitrary initial bytes.

The public domain fixes the original block size 1024, requires `N < 2^31`,
and requires representable 64-bit bases and exclusive interval endpoints.
Input-word assembly and every active output byte, including each final byte,
agree at the existing finite-pointer expressions. These are selected-domain
premises; native allocator freshness, tensor layout, alignment, provenance,
admission and endianness remain unproved.

`byteProgram` is a mathematical memory-effect projection. For a nonzero pid,
its loaded/result functions need not evaluate. Therefore this theorem does
not establish those native loads' success, faults or poison behavior. It is a
final-state representation result, not a stepwise original-Python, TTIR or
native compiler execution refinement. All dtype/error/host and concurrency
obligations outside the stated domain remain open. Three inherited ReLU
numerical callbacks remain visible.

Independent reviews checked all seven source files and exact frozen evidence.
The vector package has three successful compilations, 16 named axiom checks,
two noncircularity checks and three audited modules containing five audit
directives. The ReLU package has four successful final compilations, 30 named
axiom checks, three noncircularity checks and four module audits. Both final
batches have zero warnings. ReLU retains all 13 actual compilation attempts,
including two failed attempts with exit code 1 and their exact logs/source
snapshots. Failures are not counted as successful checks. No new frontend,
native backend, GPU or comparator execution is inferred from these records.

[root-review.json](root-review.json) binds the reviewed inputs, actual
independent review schemas and six new claims. All previous 30 declaration
records and source views are preserved. [numerical-followup.json](numerical-followup.json)
is byte-for-byte identical to its predecessor: all 20 rules remain, with no
new rules or duplicate ReLU obligations. Exact unified-queue acceptance is
still required; structural gaps cannot be deferred as numerical assumptions.

[validation.json](validation.json) records the observed registered build,
catalog, sharding tests and global gate outcomes after integration. The old
210 benchmark-tool tests are reused only after checking their unchanged
tool/test inputs. The original metadata baseline and its input hashes are
preserved. Historical/exact-51518a8c isolated comparator replays exclude these
seven modules. Source files live only at their repository source paths;
temporary caches, compiled oleans and symlinks are not retained here.

The actual registered Lean build passed all 7,322 jobs. Catalog generation
passed on its first attempt, and all 25 sharding tests passed. The global
audit passed eight gates and exited 1 at exactly three: scope review reports
183 unaccepted original programs, while comparator preflight and library
replay report the official comparator unavailable to the host runner. The
global runner did not run corpus replay. These observed partial-extension
results do not establish whole-program or native-backend acceptance.
