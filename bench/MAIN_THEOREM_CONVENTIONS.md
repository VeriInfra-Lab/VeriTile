# Main-theorem formalization conventions (`bench/tritonbench_g`)

Every selected main kernel entry must culminate in a **main theorem** (the "headline",
usually named `<kernel>_output_summary` / `..._compute_correct` / `..._general`)
that meets *all* of the requirements below. These are the standard this repo
holds every per-kernel correctness proof to; use this as the checklist when
adding, reviewing, or hardening a kernel.

For the selected TritonBench-G revisions, `Correct.lean` and `FPEquiv.lean` are
independent reader entries. Shared source belongs in `Kernels.lean`, reusable
IO data may live in `IO.lean`, and helpers required by the main proofs belong in
`Proofs/`. `Correct.lean` proves only the independent Real mathematical
specification; `FPEquiv.lean` compares two actual kernel implementations as
`original ≡[R] candidate`.
Optional single-source FP/native correctness, host proofs, Tensor applications,
candidate-correctness applications and audit entries belong in `Misc/`, for
example `Misc/NativeCorrect.lean`, `Misc/TensorBridge.lean` and
`Misc/PublicAudit.lean`. These optional entries may import reader entries;
the reader entries and their required `Proofs/` helpers must not import `Misc/`,
and supporting modules in `Proofs/` must not import `Correct.lean` or
`FPEquiv.lean`. This organization adds no per-task FP32-correctness or host-proof
acceptance requirement.
Auxiliary files do not need a `specification`. The canonical selected paths and
separate IO-format progress are recorded in `tritonbench_g/task_progress.json`;
historical acceptance or the earlier kernel/Tensor split does not certify the
current public format. Preserve existing results, including full grids and
stages, while adapting the interface.

The shared source entry must also be directly readable. Write the actual original
and existing candidate with ordinary `triton { ... }` definitions, preserving
precision, casts, branches, masks and runtime arguments. Task-local whole-program
generators, handwritten statement ASTs, dummy assignments removed before execution,
and statement-list surgery are source-organization issues even when the resulting
program has a valid proof. Keep required AST decomposition/rewrite lemmas in
`Proofs/`; do not use a source hidden in `Proofs/` as the authoritative program.
Explicitly shared source definitions and genuine multi-stage schedules are allowed.
An explicit port-metadata wrapper around an unchanged complete source body is also
allowed. A cleanup must preserve the complete selected program or prove its stated
semantic connection; it must not silently change the accepted input domain.

`Contract.lean` states admitted inputs and the independent mathematical target.
Reusable `IO.lean` describes logical coordinates, memory locations, runtime argument
bindings and invocation schedules. These descriptions do not assert the desired
output formula. Required conditions must be explicit in the main theorem or its
public IO judgment; optional Tensor/host applications remain in `Misc/`.
Use one canonical description for a shared interface. Readers may name a thin
instance selecting the Real/native interpretation, but must not copy complete
read/write/mask/schedule records already present in IO helpers. Reusable pure
IO data belongs in `IO.lean` (or an explicitly shared IO module); its execution,
safety and mathematical connection proofs belong in `Proofs/`. Preserve old
names with transparent aliases when useful, and check complete-record equality
when consolidating duplicates. Distinct encodings, channels, observers and real
multi-stage schedules may require distinct views; explain their relationship
instead of forcing them into a false common representation.
Source readability is reviewed separately from whole-task proof completion and
does not add native, host, hardware or numerical-experiment obligations.

Apply the same responsibility split inside auxiliary `Reference` modules:
channel packing and coordinate adapters belong in IO, required ordered formulas
and proof connections in `Proofs/`, and optional native results with their exclusive
adapters in `Misc/`. Classify actual declaration uses: a native formula used to
prove the main program pair remains a required helper. Reuse an existing pure
mathematical specification rather than introducing another target. A compatibility
import may remain a thin forwarder; it must not become another mixed entry.

---

## 0. Keyword — a headline is declared `specification`

Every headline is declared with the **`specification`** keyword
(`VeriTile/Meta/Specification.lean`, imported via `VeriTile.Triton`), not
`theorem`. It elaborates identically to `theorem` (same kernel object, same
axiom footprint); the keyword is the machine-readable marker the audit
tooling keys headline discovery on (`scripts/spec_sheet.py`,
`bench/check_proof_gap_manifest.py`, `bench/audit_trust_prep.py` — no
name-suffix heuristics). Helper lemmas stay `theorem`; do **not** declare
internal lemmas as `specification`.

The main public meanings, expressed without changing this keyword, are
`Spec.Real claim` for real-valued correctness against the mathematical formula,
and `lhs ≡[R] rhs` (`open scoped VeriTile.Spec`) for implementation equivalence
under explicit rewrite rules. Distinguish formally proved local rules from
empirically admitted numerical assumptions as described in §4d.
Source-to-native observation results retain their native IO judgment as ordinary
auxiliary theorems in `Misc/`, for example `Misc/NativeCorrect.lean`. They do not
replace a two-program equivalence headline in `FPEquiv.lean`. If a task has no specified
implementation pair, record that missing target rather than inventing a pair
or counting the observation result as completed implementation equivalence.
The model `R` bundles the rule table and its admission assumptions; keep
experiment configuration and result records out of the headline's parameters.
Use `#print_spec` to expose assumptions, primitive/rule dependencies and evidence.
Existing unwrapped declarations keep their mathematical/abstract-rounding
meaning; `⊨[R]` and `≡[R]` are not automatically FP acceptance certificates.

## 1. Position — the last theorem is the main theorem

The **final theorem in each main reader file must be the headline** (declared
`specification`, per #0). A reviewer should be able to read the last theorem,
then walk *up* to the lemmas it references, and see the whole trust chain. No
pinned corollaries, dead helpers, or stray lemmas after it.

Keep one headline per selected kernel or independently required stage/branch.
Forward/backward kernels, producer/consumer stages and genuine source variants
may therefore have multiple headlines in a reader file. Do not collapse these
results to satisfy a syntactic one-declaration count.

Keep a complete multi-stage main result when the selected computation requires
that schedule; do not demote its only complete result or invent stage goals just
to reduce the number of declarations. A direct Real proof uniform over an
original/candidate source family is also allowed. An additional application
combining repeated Python calls, beyond an existing kernel headline, belongs
in `Misc/` (for example `index_select_bwd/Misc/CompositionCorrect.lean`).

Source-identity lemmas, candidate Real-correctness corollaries, and a native
observation supporting an existing implementation-equivalence headline are
auxiliary results. Preserve them as ordinary `theorem`s in named supporting
files, rather than additional reader-entry `specification`s. Helpers required
by a main proof stay in `Proofs/`; optional applications belong in `Misc/`.
A task whose only FP result is a source-to-native observation retains that auxiliary proof and
records that its implementation-equivalence target is missing. Supporting `Proofs/`
modules must remain independent of reader entries. An optional auxiliary file
that applies a public result may import it under `Misc/`, like
`Misc/SourceBinding.lean` or `Misc/CandidateCorrect.lean`.
The same distinction applies to proof aliases declared with `alias` or `abbrev`;
changing the keyword alone does not make an auxiliary conclusion reader content.
Named source programs, IO views and independent formulas may remain in readers.

Run `python bench/check_main_entry_roles.py` to compare all 183 selected readers
and re-exports with the explicit per-declaration role review and check that their
repository import closures exclude `Misc/`. Reader changes require a new role
review. This guard does not infer semantics from declaration counts, re-prove
Lean results or certify a single combined build generation. Historical optional
specifications already isolated in `Misc/` are outside the main-entry inventory.
Run `python bench/check_io_organization.py` for the separate canonical-IO review:
it checks current selected paths and reviewed source hashes, and rejects detailed
IO records returning to readers. Semantic sharing and genuinely distinct views
are recorded explicitly in its per-task report, not inferred by the script.

## 2. Quantification — general mathematical proofs, scoped numerical evidence

The dimension-general rules below apply to the mathematical/legacy corpus.
For atom-based `≡[R]`, concrete shape/stride/precision/backend/probe
configurations belong to the atomic assumptions. Report their checked scope;
evidence for one shape cannot justify admission for another. Structural and algebraic helper theorems should remain parameterized
where possible. A theorem quantified over configurations with an explicit
validated-evidence premise is conditional, not evidence that all configurations
passed. Pending records are not acceptance results.

- The headline is **universally quantified over all shapes / strides /
  compile-time flags** as symbolic `Nat`/`Bool`/`ℝ` parameters. It must **not**
  be pinned to the benchmark's concrete test dimensions.
- **Zero test-shape declarations.** Pinned per-shape / per-flag instantiations
  (the tell-tale marker is the token `python` in a decl name — e.g.
  `*_python_case4_*`, `*_python_block128_*`, `*_python_128x128_*`,
  `*_python_test_shape_*` — anything that is *not* a `_general` theorem) are
  redundant given the general headline and must be removed, along with any helper
  lemmas that become dead once they are gone.
- A genuine compile-time *case* decomposition (one theorem per autotune/flag
  branch) is fine **only** when each case theorem is itself dimension-general
  (e.g. `..._python_caseN_output_summary_general`); the `python`/`caseN` in the
  name then denotes the source kernel / flag branch, not a test shape.

## 3. Genuine — non-self-referential

The `expected` value (what the theorem claims the output is) must be a
**formula over logical inputs** (arrays, a `Finset.sum`, a pure
`Math/*` spec, …). It must **never** be `exec(...).readMem …` — i.e. the kernel's
own executed output read back and compared to itself. `scripts/spec_sheet.py`
must report `self-ref-flagged: 0`.
Operational input-memory versions of the same formula remain useful helper
lemmas, but do not belong in the selected public headline.

### Reader-visible mathematics and exact scope (2026-10-22)

The selected `Correct.lean` entry must show the mathematical result being
proved, as in `Spec.Real (io ⊨ fun inputs index => mathematical_expression)`.
Simple formulas belong directly in the headline. For a genuine multi-output
or multi-stage result, a pure mathematical definition displayed immediately
beside the headline is appropriate. Merely referring to an undisplayed
`reference`, `target` or `formula`, or eta-expanding that name, does not meet
the reader standard. Reuse the canonical specification and prove the displayed
definition equal to it; do not maintain an unrelated second target.

For ordinary matrix inputs and row outputs, expose logical coordinates such as
`xs row col`, without a single-channel number, flattened memory address or its
bound proof in the main formula. Keep the bijective coordinate maps and actual
schedule in `IO.lean`, and prove equivalence to the complete existing judgment
in `Proofs/`. A logical scheduled IO may carry the schedule behind ordinary
`⊨`; retain the explicit coverage/order premise in the headline. Reindexing
must preserve every input, output, execution, safety and frame obligation.
`LogicalScheduledKernelIO` supplies this view for existing scheduled regional
Real contracts; it does not define a new executor or a weaker correctness test.

Named accessors over raw `ChannelValues` or flat storage do not by themselves
supply a logical tensor interface. Use actual named mathematical input/output
types and prove the complete transport to the previous judgment. Genuine
multi-output, mixed-type and multi-stage formulas retain their structure;
channel packing belongs in IO. With arbitrary strides, aliases or unused
suffixes, a graph of mathematical arrays, complete backing values and their
connection equation can retain a bijection. Those arrays are constrained views,
not arbitrary independent inputs. An existing logical row `Fin B → ℝ` needs no
new record merely to move the active `Fin n` embedding into an IO accessor;
keep the complete B-wide domain and judgment.

Keep a task's memory descriptor and its logical input/output view together in
`IO.lean`, using namespaces such as `IO` and `Logical`. Do not create a separate
task-level `LogicalIO.lean` or `LogicalCoordinates.lean` just to hold that view.
The complete judgment equivalence and execution connections remain in `Proofs/`.
Moving these definitions preserves their public names, bodies and full judgment;
resolve import cycles by separating the necessary pure data from proofs. Genuine
independent stages or encodings may keep explicitly justified interface modules;
shared Logical IO library types remain shared.


Logical dimensions and implementation tile dimensions are distinct. A complete
`n`-column softmax sums over `n`; a padded `B`-lane proof connects to that result
under the explicit condition `n ≤ B`. Keep the older window theorem when it has
a useful more general scope. The same check applies to reduction axes, tails,
branches, configuration coverage and actual full-grid or multi-stage schedules.
A one-program result must say so. Do not turn a complete staged result into a
single-stage claim to make its statement shorter.

Keep every required domain and memory premise in the headline or its documented
public judgment. Existing common IO namespaces provide `io ⊨{P} f`, exactly
`ImplementsOn io P f`, for a logical input predicate `P`. These braces do not
denote FP rewrite rules or a rounding model. Accurate specialized cell, integer,
observer, relation, scheduled, 3D and backend-parameterized judgments may retain
their explicit names and arguments; never replace them by a weaker ordinary
`Implements` solely to make the notation uniform. Logical indexing adapters can
remain in IO, required equality/execution bridges in `Proofs/`, and optional
Tensor/host applications in `Misc/`.

Python assertions, input contracts and theorem premises must have a documented
correspondence. Coverage already implied by launch/configuration selection is
proved from that selection; it does not require a new Python input rejection.
The per-task review records the actual formula, input domain, complete scope,
proof connection and FP program/rule binding. Format repair and a substantive
proof gap are different findings. Syntax checks, successful compilation and
earlier task acceptance alone are insufficient evidence of semantic review.
Run `python bench/check_main_formula_review.py` to check the current selected
readers, reviewed source connections, recorded rules and per-task evidence
against that explicit review. This is a drift guard, not a replacement for
semantic review, Lean proof checking or numerical validation.

## 4. Structure — a **named** surface, `⊨` preferred

The headline must be stated through a named trust surface, never as a
hand-rolled `exec` statement. There are two, in order of preference.

### 4a. The KernelIO `⊨` face (preferred)

```lean
open scoped VeriTile.Triton.KernelIO₂ in
specification <kernel>_correctness … :
    <kernelIO> … ⊨ fun xs ys j => <formula over logical inputs>
```

`⊨` (and its rounding sibling `⊨[R]`, four-hole `⊨[R, outDType]`) is the
flat-pointer-memory Hoare triple defined in
`VeriTile/Triton/Memory/KernelSpec.lean`: termination, every declared output
cell, **and the frame** — no cell outside the write set changes. The IO record
carries the addressing (windows, masks, tile length, output dtype), so the
statement stays one line. Prove it with the skin's `Implements.intro` /
`ImplementsR.intro`, which reduces the goal to `FlattenOk` + `TraceSafe` +
the region-model Hoare triple. See
[`documents/CorrectnessSurfaces.md`](../documents/CorrectnessSurfaces.md).

Pick the skin whose *signature shape* matches (inputs/outputs, program axes,
metadata scalars, gather index). If none fits, say so in the preamble rather
than bending the kernel to the skin.

Regional Algorithm IO (`Memory/RegionalKernelIO.lean` and its skins) preserves
read-only input aliases without imposing the flat allocator's disjointness.
Structural source and native-observation entries use `Float/ProgramIO.lean`:
typed logical channels, explicit runtime register bindings and actual launch
schedules. Its judgment proves actual source trace safety, successful execution,
typed observations and an exact raw-cell frame. It does not imply flat-pointer
transport. `ProgramEquivalence` compares two actual schedules on the same IO
signature. Explicit read-result and exact-cell input variants preserve the
originally proved observation/binding scope.

For existing arbitrary-array gather results, `Float/ArrayProgramIO.lean` binds
named typed arrays and declares a finite logical access footprint. It does not
claim infinite backing: bounds cover only those accesses, and actual source
safety must be proved. Successful execution, outputs and the exterior frame are
required even without a bounds witness. `Float/FillProgramIO.lean` and
`Memory/AlgorithmFillProgramIO.lean` admit only concrete source stages and
explicit finite typed fills, preserving their actual order and successor
memory. These interfaces do not accept custom runners or safety callbacks.

`Float/OrdinaryProgramIO.lean` retains an ordinary grid in which each source
invocation reads the same initial state. It proves actual invocation safety
and success, disjoint source write footprints, their merged outputs and the
exact exterior frame. It must not be substituted with a serial schedule when
the original proof uses independent programs. The regression test executes a
producer and consumer to distinguish these two memory semantics.

`Memory/ParameterizedAlgorithmProgramIO.lean` adds typed runtime arguments to
the existing Algorithm stage interface. A fixed sequence of register bindings
prepares the initial state once; the original source still reads its runtime
registers. Ordinary CTAs share that prepared predecessor. Access safety follows
the actual prepared state, while input binding and the exact memory frame refer
to the original state. Empty arguments recover `AlgorithmProgramIO` exactly.

`Float/MaterializedProgramIO.lean` preserves native models that finish stored
scalar values after each CTA before the next CTA reads them. It uses the actual
Structural invocation, a type-preserving scalar codec, explicit ABI arguments
and fixed fresh local state. Access safety follows the materialized successor
memory; output binding and the exterior raw-cell frame remain proved results.
Delaying materialization is allowed only with a separate equivalence proof.

Typed fused multiply-add uses `ComputeExpr.fmaTyped` and the matching optional
dtype-selected primitive in `Structural.Algebra`. All three typed operands and
both broadcasts are preserved, with one fused primitive call per lane. Missing
support fails; it is never replaced with separate multiply and add. Access
safety checks every operand even when an output store is masked. The legacy
FMA API is unchanged. These are interpreter properties, not empirical
floating-point accuracy or GPU-conformance assumptions.

`Float/IntegerDotIO.lean` provides the existing contextual integer-dot source
semantics with explicit scalar topology and fixed-arity native instruction
ports. `IntegerDotPerAddressProgramIO` preserves independent atomic orders at
each output address: every complete source launch must execute safely and
successfully, and source locality, event coverage, exact cell traces and the
frame are proved conclusions. It does not introduce a general concurrent GPU
executor or an arbitrary whole-dot callback.
Its explicit `accumulatorNat` profile recognizes the original accumulator
update and requires the original Nat coordinate registers. Coordinates are
not reduced modulo matrix dimensions; the actual carry retains the existing
integer encoding normalization. The default `scalarCast` profile is preserved.
The reset variant initializes finite typed output cells, then proves input
preservation, source execution, locality, coverage, results and the original
exterior frame. Each task must connect this initializer to its source reset.

`Concurrent/EventProgramIO.lean` covers the existing source-decoded concurrent
tasks using closed event tapes or finite controllers. Scalar operands are
expressions interpreted on the current memory; CAS returns its old word and
the following branch sees that returned value. A finite schedule must reach
`Done`; there is no fairness or all-schedules claim. Each task must expose its
exact source-decoder and execution refinement, prove a concrete capacity
witness, and retain all source reads, including reads whose output store is
masked. The shared judgment requires successful completion, safety of the
whole event trace, finite outputs and exact exterior frames. Locality lemmas
transport the same safe execution to arbitrary matching external memory.

Public IO is data plus a shared judgment, not a renamed task-local whole
correctness proposition. Keep source/ABI, channel and schedule choices
inspectable. Put access/runner/frame assembly in supporting proofs so the main
proof is a short composition. Optional flat corollaries and checked-Tensor
applications are ordinary auxiliary theorems in `Misc/`.

### 4b. Legacy `ComputeCorrect.Realizes_without_Rounding`

This surface remains part of the historical corpus. For selected current
entries, a raw initial-state/write-map interface still needs a logical IO
adapter; it is not an exemption from the current reader format.

```lean
ComputeCorrect.Realizes_without_Rounding
  (kernel := <faithful surface>)
  (initialState := s)
  (write := <WriteMap: which cells to check>)
  (expected := <input-memory closed form>)
```

A multi-output kernel is stated as a **single headline theorem whose conclusion
is a conjunction of `Realizes`**, one conjunct per stored output (scalar outputs
use `fun _ : PUnit => some (region, offset)`; masked tile outputs use
`ComputeCorrect.WriteMap.writeIf mask addr`). Keep the conjunction — do **not**
split the outputs into separate top-level theorems; the file ends on one bundled
headline.

### 4c. What is not allowed

Do **not** leave the summary in a raw `exec` form — neither
`(hExec : exec … = some s') → s'.readMem … = expected` nor the existential
`∃ sF, exec … = some sF ∧ <output readback>`. Both are proof artifacts, not
public surfaces, and the existential one additionally tends to drop the frame.
(`Realizes` and `⊨` internalize the `exec` quantification, so the `s'`/`hExec`
binders drop out of the statement.)

`bench/audit_tritonbench_g.sh`'s correctness-surface scan enforces this. A port
that genuinely cannot meet it yet must carry a
`Correctness-surface blocker:` preamble marker **and** a row under
"## Correctness-Surface Blockers" in `proof_blockers.md`; the scan rejects an
unregistered offender and a stale registration alike.

### 4d. Floating-point implementation-equivalence headlines

State these as `lhs ≡[R] rhs` in the `VeriTile.Spec` scope. `R` supplies the
rule table and its admission assumptions; `Spec.FloatingPoint` is the underlying
relation. An empirical numerical atom uses an `AcceptedAtom` obligation backed
by its applicable two-gates record. Reuse an existing applicable admission;
neither a new kernel nor correlated operands alone demand another experiment.
Never assume the whole equivalence as a shortcut. Lean checks the derivation and the
unchanged implementation signature. Symmetry, transitivity and common sequence
context are formal rules under those assumptions, not empirical gate guarantees.
Correctness against a mathematical formula stays in the real model. Print all
premises and atom records; do not equate admitted atoms with IEEE bit equality
or claim that an unrun checker supplied a validated artifact.

Source-to-native observations remain auxiliary results through a shared native
IO judgment in `Misc/`; they are not `FPEquiv.lean` headlines. Required local
connections used by the main equivalence proof stay in `Proofs/`.
Reuse the task's existing implementation pair where available, keeping its concrete transformation
visible. Do not invent a second kernel or a floating-point rewrite to obtain
`≡` syntax. Native
primitive/model boundaries remain distinct from transformation assumptions;
elementary mathematical facts and program proof gaps are not deferred numerical
experiments.

The refined FP policy (2026-10-21, root `AGENTS.md`) separates three layers:

- The mathematical `Correct` result uses independent Real formulas and Mathlib.
  Required integer and quantized formats remain explicit. An exact format
  conversion with an independently proved mathematical decoding formula is a
  valid result; a rounded implementation expression alone is not a substitute
  for the independent formula.
- The main kernel FP proof uses typed operations, proved local rules or explicitly
  admitted numerical assumptions, and structural reasoning. A main result may
  use a concrete FP model and its strict lemmas directly; it need not be
  parameterized over arbitrary opaque operations. Keep dtype, casts, FMA versus
  separate operations, order and source/output/frame connections visible.
  Concrete conversions, comparisons and explicit profile bindings alone are
  neither defects nor mandatory interface-refactoring tasks.
- The numerical/model layer retains concrete software models and exact local
  proofs. Generic scalar facts such as concrete multiplication commutation belong
  in `VeriTile/Triton/Float/`; task-specific formulas and index facts stay in
  `Proofs/`. Preserve compatibility aliases when extracting shared laws. Do not
  duplicate proofs or introduce a new general framework just to move one lemma.

Record rule evidence explicitly:

1. **Formally proved:** name the theorem, model, format, configuration and domain.
   A kernel may use the strict theorem directly or consume an abstract local law
   instantiated by it; abstraction is optional, not an acceptance gate.
   No empirical `AcceptedAtom` should be fabricated or required
   for the same already-proved fact. Model soundness does not itself establish
   hardware conformance.
2. **Empirically admitted:** name the numerical rule, applicable admission and
   recorded precision/domain/distribution/configuration/criteria. Reassociation
   and FMA contraction need not be exact identities; passing an experiment does
   not make them unconditional IEEE equalities. Pending evidence remains pending.
   A rule used through existing admitted `R` remains conditional even if a
   similarly named theorem exists for a separate concrete model.

Primitive/profile conformance, source constants, finiteness predicates, layout
and successful execution are not empirical rewrite rules. Purely structural
changes need no numerical assumption. Exact proofs of approximate error bounds
also count as formal evidence. The existing conditional `≡[R]` derivation does
not by itself supply a whole-kernel error bound or compose statistical gates;
this policy does not redefine that judgment. Track actual rule-evidence gaps,
optional refactoring and numerical experiment deferrals separately, preserving
accepted results and their scopes. Integer/quantized APIs retain their independent
integer, wrap and conversion formulas; do not demand an unbounded ideal-Real
claim from a result that explicitly proves a different supported specification.

## 5. Axiom-clean

`#print axioms <headline>` must be a **subset of `[propext, Classical.choice,
Quot.sound]`**. No `sorry` / `admit` / `native_decide` / `ofReduceBool`, and no
`sorryAx` anywhere in the transitive dependency.

## 6. Honest hypotheses and honest scope

- Real preconditions (region distinctness / no-aliasing, offset injectivity,
  `hundef`, positivity of block dims, …) are stated as **explicit honest
  hypotheses** of the headline — not silently assumed or worked around.
- Modeling boundaries are stated plainly (host launch / grid / autotune are the
  *trusted boundary*; arithmetic is over `ℝ`, not bit-accurate IEEE; casts erase
  to identity at the algorithm layer; …).
- If the model genuinely cannot cover a path (e.g. a faithful `tanh(-∞) = -1`
  makes a masked-lane argument fail), **narrow the theorem's scope and document
  it** — do not fake a spec that does not hold. Prefer *genuine-but-scoped* over
  *pinned-but-annotated*.

### Never lie, never silently ignore

If a kernel (or a path/output of it) **cannot be dimension-generalized or proved
genuinely**, that is a **reported failure — surface it as an explicit ERROR /
blocker**, with the exact goal or the exact obstacle. It is **forbidden** to:

- **mis-report** it — claim a theorem is general / genuine / axiom-clean when it
  is not (e.g. calling a pinned theorem `_general`, or an `exec`-readback spec
  "genuine");
- **silently ignore / drop** it — leave the shape pinned with no signal, quietly
  weaken the spec, or omit the output from the summary without saying so;
- **paper over** it with `sorry`/`admit`/`native_decide` or a vacuous/trivially-
  true statement.

The honest outcomes are exactly two: **(a)** a headline meeting §1–§6 in full, or
**(b)** a clearly-reported blocker (the specific goal that won't close, or the
specific reason generalization is impossible) plus, if partial progress is
landed, an explicitly-scoped theorem whose narrowed scope is stated in its own
hypotheses/docstring. Partial-but-honest is acceptable; anything that reads as
"done" when it isn't is not.

## 7. Shared pure math lives in `VeriTile/Triton/Math/*`

When a genuinely reusable pure-math object exists (e.g.
`crossEntropyLoss = stableLSE − logit_target`, `klDivSpec`), put the **pure
function** in `VeriTile/Triton/Math/*` and connect each kernel with a **bridge
lemma** proving its (kernel-coupled) spec reduces to the pure core in the base
regime. Kernel-coupled specs — those taking `BlockState` / `RegionName` / program
ids — **stay in the kernel file**; only pure `ℝ`-level math is lifted.

## 8. Tidy

- **No dead code**: delete superseded boundary lemmas (e.g. store-slice copies
  once the genuine forward subsumes them) and any helper left unreferenced.
- **Docstrings match the code**: the module's proof-architecture diagram and
  prose must name only declarations that still exist and describe the current
  headline form. No stale references.
- Regenerate spec sheets (`scripts/spec_sheet.py`) after changes;
  `self-ref-flagged: 0` and `no-summary: 0`.

## 9. One task owns its source and proof directory

Keep the task mapping with the benchmark's Python kernels. Do **not** merge two
distinct benchmark tasks into one directory even if their proofs are near-identical
(e.g. `cross_entropy` vs `cross_entropy_ops`). Reuse common supporting proofs;
keep each task's selected `Correct.lean` and `FPEquiv.lean` independently readable.

---

## Verification recipe

Bench files are checked individually (they are not `lake` library targets):

```bash
# 0. build the shared library once (so imports resolve)
lake build VeriTile
lake build VeriTile.Triton             # umbrella prelude (what ports import)

# 1. compile a kernel file — must be exit 0 with zero `error:` lines
lake env lean bench/tritonbench_g/<kernel>/<File>.lean

# 2. no cheats anywhere in the file
grep -nE '\bsorry\b|\badmit\b|native_decide' bench/tritonbench_g/<kernel>/<File>.lean   # -> none

# 3. axioms of the headline — append, run, then remove the line
#    (use the FULLY-QUALIFIED name; #print after `end <ns>` needs it)
#    expect: [propext, Classical.choice, Quot.sound]
printf '\n#print axioms VeriTile.Bench.TritonBenchG.<Ns>.<headline>\n' >> <File>.lean
lake env lean <File>.lean | grep -A2 'depends on axioms'
git checkout -- <File>.lean   # or manually delete the appended line

# 4. spec sheet: no self-reference, has a summary
python3 scripts/spec_sheet.py bench/tritonbench_g/<kernel>/<File>.lean
```

Notes:
- Verify with `lake env lean` (the LSP/`lean-lsp` may be unreliable in some
  environments); do **not** run bare `lake build`/`lake clean`/`lake exe cache`
  while iterating on a kernel — they can invalidate the prebuilt oleans.
- Masked-reduction / log-sum-exp proofs need
  `import VeriTile.Triton.Semantics.{TiledIndexing,MaskedReduction}` +
  `VeriTile.Triton.Math.LogSumExp` and **`open VeriTile.Triton.TiledLogSumExp`**
  (the lemmas `validLanes` / `partialLSE_full` / `sup'_masked_map_eq` / `stableLSE`
  live in that nested namespace).

---

*One line:* the last theorem is a **dimension-general, non-self-referential,
axiom-clean** headline using the named shared IO or equivalence judgment
appropriate to the accepted result, with honest hypotheses; pure math is factored into `Math/*`; the file
carries no dead code or stale docstrings — and if any of this genuinely cannot be
achieved, that is **reported as an explicit blocker, never faked or silently
ignored**.
