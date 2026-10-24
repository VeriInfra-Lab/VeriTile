# fp64 logical grids, dtype admission protocols and ReLU checked byte access

This batch adds 13 Lean modules and seven public specifications across two
tasks: three execution contracts and four protocol contracts. The cumulative
supplement contains **39 execution contracts and four protocol contracts across
10 tasks**. Whole-original-program acceptance remains **0/183**, with all 183
pending. All original stages, dtypes and branches remain required; only precisely
registered numerical correspondence may be deferred.

The binary64 foundation uses actual `BitVec 64` words, complete field decoding
and typed scalar casts/addition. The new task-local memory/grid interpreter has
optional, fully tagged cells; absence and wrong tags never become default words.
It performs actual logical stores and proves an independent full-memory result
and frame for every complete sequential program permutation.

`runLanes` computes and writes one lane at a time. All reads use the fixed
program-entry snapshot, while stores update working memory. This is not a
load-all/compute-all/store-all instruction trace or a source/compiler refinement.
Output/input separation is an explicit selected-domain premise. Equal input
names require complete tag/payload compatibility. Masked lanes need neither
input cells nor defined conversions. Errors distinguish unavailable input from
undefined conversion and return no final memory; native poison, traps, partial
writes and rollback behavior are not established.

All 29 successful fp64 operand profiles have logical-grid results. Twenty have
floating outputs and are automatically defined under the chosen total callbacks;
nine integer-output profiles retain an active conversion-domain premise. The
previous 160 profiles use the shared interpreter and concrete software
operations. These 29 use a separate task-local interpreter and 15 unverified
numerical callback families. **The combined 189/217 inventory spans two models**;
it must not be reported as 189 profiles proved in one interpreter. The remaining
28 promoted fnuz pairs have no successful logical grid.

The four dtype-admission specifications are finite protocol contracts. Their
selected sm90 model partitions 289 canonical ordered pairs into 189 gate-pass,
64 signature-stage rejection and 36 addition-stage rejection outcomes. Source
unit checks and frozen frontend observations support this classification; a
formal original Python/JIT/compiler execution or exception relation remains
open. Each record has `model_kind = role = protocol_contract`, empty covered
stages and `scope_relation = not_an_execution_contract`. These declarations add
zero execution contracts, value profiles or completed original programs.

The ReLU refinement adds checked partial byte accesses to the existing nine
integer/bool profiles. Its mathematical memory has an eight-bit payload and
explicit allocated/readable/writable maps. Every active input byte of every pid
passes an ordered failing-access bind before the pid-zero store gate. A nonzero
pid with an unavailable first active input byte fails, even though the older
total memory-effect projection has no output effect for that pid.

All N input words must be readable; only the first `min(N,1024)` output words
must be allocated and writable. The unmodified output tail needs no allocation
or access permission. Complete whole-program permutations succeed on the memory
left by prior programs and agree with the old byte grid, independent byte
specification, final tagged logical view, typed result and frame. The inherited
finite-address certificate remains subject to its explicit bounds and separated
interval premises. Native allocation, access rights, layout and provenance are
not proved by these mathematical maps.

On failure, the ReLU model retains the exact preceding mathematical writes;
this establishes no native fault or poison order. Nonzero-pid scalar callbacks
are not separately evaluated. Zeros outside loaded prefixes only complete a
mathematical container and do not specify original no-other masked loads.
Native instruction interleavings, source/compiler correspondence, host/module
composition, other dtypes and error branches remain structural obligations.

All 20 previous numerical-rule records are preserved exactly. Fifteen fp64
family descriptions bring the total to 35. The separate exact candidate file
binds 73 dtype/port instances and 120 applicable public declaration/port pairs
to callback names, source, raw TTIR, configuration and statement fingerprints.
Every candidate remains unvalidated. Descriptive records and candidates do not
constitute accepted unified queue scope bindings. ReLU retains its three old
callback labels at its new statement; its two existing descriptive rules are
not duplicated. Admission adds no numerical rule.

Independent reviews cover all four frozen packages. The final producer runs
have 13 successful module compilations in total, 101 named axiom checks, ten
noncircularity checks and 13 module audits, with zero final warnings. The fp64
grid retains 12 actual attempts including four failures; admission retains four
attempts including two failures; ReLU access retains 21 attempts including seven
failures. Core retains its three final compilation records. No failed attempt
is reported as a successful check, and no new native/GPU execution is inferred.

Evidence-only admission Lean snapshots/probes retain their original bytes under
`.lean.txt` paths, with original producer paths in the evidence index. Source
modules occur only at their repository source paths. Compiled oleans, caches and
symlinks are not copied. One adjacent `.gitattributes` rule preserves original
trailing blanks in a single frozen raw Python fragment, with the original failed
check and successful recheck retained.

[root-review.json](root-review.json) binds the four reviews and all seven new
claims. [catalog-supplement.json](catalog-supplement.json) preserves every prior
declaration record and source view. [numerical-followup.json](numerical-followup.json)
records the 35 descriptive rules and their unresolved queue status.
[validation.json](validation.json), produced only after actual integration
checks, records observed commands, exits, source/catalog preservation and gate
outcomes. Original metadata-baseline files and inputs remain unchanged.
Historical isolated comparator replays exclude these 13 modules.

## Observed integration validation

The registered eight-target Lake build exited 0 with 7335 observed jobs. Catalog generation and all 25 sharding tests exited 0. The resulting catalog contains 184 tasks, 936 public declarations, 3543 task modules and 3974 registered modules. All earlier 36 supplemental execution records were preserved; the new inventory is 39 execution contracts plus 4 protocol contracts across 10 tasks. The prior 20 numerical rules were preserved exactly and 15 unvalidated fp64 descriptions were appended.

The global audit actually exited 1. Its failed gates are `scope_review` (all 183 original-program scopes remain pending), `comparator_preflight` (the host runner requires an available official comparator) and `library_replay` (the same comparator availability preflight). All other recorded gates passed. This is a validated partial extension; whole-original acceptance remains 0/183. No comparator success or native numerical validation is inferred. Exact commands, observed exits and log hashes are recorded in [validation.json](validation.json).
