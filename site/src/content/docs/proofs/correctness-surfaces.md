---
title: "Correctness Surfaces"
---

## Public specification meanings

**Mathematical correctness uses Real. Implementation equivalence retains source
precision and states any required floating-point atom assumptions explicitly.**
Both retain the `specification` keyword.

| Public relation | Meaning |
|---|---|
| `Spec.Real claim` | An implementation satisfies its mathematical formula in ideal real semantics |
| `lhs ≡[R] rhs` | The two implementations are equivalent under the two-gates-admitted atomic rules supplied by `R` |
| Native IO judgment | Actual source execution matches the declared native observation, with its explicit primitive model; no transformation or native-to-Real equality is implied |

Selected TritonBench-G entries use a genuine shared IO judgment with an
independent logical formula. `IO.lean` may hold shared source, argument, channel,
window, mask and schedule data; `Correct.lean` and `FPEquiv.lean` remain independent
short reader entries. Operational assembly and optional flat/Tensor corollaries
belong in supporting files. The current format rollout is separately counted in
the [task checklist](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/tritonbench_g/README.md).

For the selected main readers, display the actual independent mathematical
formula in `Correct.lean`: simple formulas directly in
`Spec.Real (io ⊨ fun inputs index => ...)`, and genuinely complex multi-output
or staged formulas as pure definitions immediately beside their headline.
An undisplayed `reference`/`target` name or an eta-expanded alias is insufficient.
Preserve the canonical formula through a proved equality or semantic connection.
Logical dimensions must match the contract: for example, a complete `n`-column
softmax needs a sum over `n` and an explicit covering-tile condition `n ≤ B`.
Single-program, full-grid and real multi-stage scopes remain distinct.

Ordinary matrix/row interfaces should expose `xs row col` rather than a flat
channel number, address arithmetic and its bounds proof. A logical view keeps
bijective input/output coordinates and the actual schedule in `IO.lean`, with
a proved equivalence to the full previous judgment in `Proofs/`. The shared
`LogicalScheduledKernelIO` carries such scheduled regional Real contracts:
ordinary `⊨` retains their safety, successful execution, outputs and frame.
Schedule coverage/order conditions remain explicit in the main theorem.
The logical interface must carry actual named tensor axes/channels, rather than
only rename raw storage accessors. For noninjective layouts, retain complete
backing data and the mathematical-view connection as a bijective graph. The
full old/new judgment equivalence preserves its domains, masks, schedules,
observations and frame. See `bench/MAIN_THEOREM_CONVENTIONS.md` for the precise
reader rule and the existing-logical-row exception.

At task level, the memory descriptor and its mathematical input/output view live
together in `IO.lean`, under separate namespaces when useful. A logical view alone
does not need another `LogicalIO.lean` or `LogicalCoordinates.lean` file. Full
judgment and execution connection proofs stay in `Proofs/`; shared library types
and genuinely independent stage/encoding interfaces retain their own modules.


Common Algorithm, Prepared/Parameterized/Fill Algorithm, Structural Program,
Read, Array and Reduction IO namespaces provide `io ⊨{P} f` for the existing
`ImplementsOn io P f` judgment. The braces expose a logical input domain; they
are distinct from FP rule/rounding `[R]` notation. Ordinary `⊨` does not silently
remove a required domain. Specialized cell/reset, integer observation, output
relation, scheduled, 3D and backend-parameterized judgments can retain their
accurate explicit method names with visible mathematics and premises. These
notation choices do not change memory binding, execution, safety or frame.
See [the detailed main-entry rules](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/MAIN_THEOREM_CONVENTIONS.md).

The shared [regional Algorithm IO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Memory/RegionalKernelIO.lean)
and [Structural ProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ProgramIO.lean) retain input
binding, actual source-access safety, successful execution, output observations
and an exact exterior cell frame. Structural launch plans contain actual kernels,
primitive algebras, typed runtime argument bindings and program IDs; they cannot
replace execution with a task-provided callback. Source precision stays in the
selected AST and declared primitive model. Regional IO does not automatically
provide flat-pointer relocation, allocation separation or byte-level transport.

[RegionalKernelIOOn](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Memory/RegionalKernelIOOn.lean) adds an
explicit predicate on logical input values to the regional judgment. It keeps
the same source-safety, execution, output and frame conclusions; the domain
cannot supply an execution-state predicate or replace those proof obligations.
The signed-length [copy interface](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/repairs/tritonbench_g/var_len_copy/Correct.lean)
uses this to constrain starts only when the input length is positive.

[ProgramEquivalence](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ProgramEquivalence.lean) compares two
complete actual launch schedules on the same IO layout. Both must succeed and
be source-safe. [ReadProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ReadProgramIO.lean) keeps
decoded-result windows separate from writable windows; untouched result cells
remain in the raw frame. [InputBindingIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/InputBindingIO.lean)
provides explicit exact-cell input bindings when that is the originally proved
domain. These choices preserve existing input/observation scope; they do not
introduce numerical assumptions. See [Lion](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/repairs/tritonbench_g/adam_update_triton/Correct.lean)
for runtime scalars and two in-place outputs.

[AddressedInputsIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/AddressedInputsIO.lean) retains source
region and offset when the existing native model labels its input terms by
address. [ObservedProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ObservedProgramIO.lean)
allows arbitrary symbolic inputs with separately typed native outputs: a pure
output observation composes with the source theorem without changing its input
domain, source safety, execution or frame. Neither interface supplies an
execution callback. Integer metadata and cell dtype tags remain unchanged.

[ObservedProgramEquivalence](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ObservedProgramEquivalence.lean)
compares the native observations of both successful source schedules while
preserving arbitrary symbolic inputs and both raw frames.
[RelationalProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/RelationalProgramIO.lean) supports
independent logical output relations, such as a maximum and its least index.
Its output witness must bind to the actual source result; a related answer
alone does not satisfy the judgment.

[IntegerStageIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/IntegerStageIO.lean) uses the existing
llrint/integer-storage evaluator, promoted unchanged from the quantizer proof.
Its safety follows that evaluator's actual successor states, including stored
integer narrowing. [IntegerOutputIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/IntegerOutputIO.lean)
separately exposes signed or unsigned fixed-width integer observations where
the original result already used them. An output view changes neither raw
storage nor execution, and it preserves dtype tags and non-integer payloads.

[RegionalScheduledKernelIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Memory/RegionalScheduledKernelIO.lean)
reuses regional Algorithm channels for finite schedules of the same source.
Launch data changes only selected program IDs and optional grid dimensions.
Each invocation sees its predecessor's actual writes. Successful outputs and
the raw frame are unconditional on a capacity witness; declared capacities
separately imply actual source-access safety.

[AlgorithmProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Memory/AlgorithmProgramIO.lean) supports
different source kernels in successive stages. A stage is one actual launch or
the existing ordinary grid relation, whose programs read the same initial
memory and merge disjoint writes. Each later stage receives the preceding
stage's actual result. Safety, outputs and the raw frame refer to that same
execution witness. A closed initializer may preserve or zero `undef` once;
it cannot change memory or registers. This regional interface uses the existing
nonrounding `Kernel.TraceSafe`; it does not identify it with `TraceSafeR .triv`
or supply the separate no-underflow obligations of a flat-pointer bridge.

[PreparedAlgorithmProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Memory/PreparedAlgorithmProgramIO.lean)
handles existing sequential source models that clear registers or zero `undef`
at each launch. These are closed Boolean choices; preparation preserves memory,
and each invocation executes the fixed source interpreter on its actual
predecessor. Success, outputs and the frame remain unconditional on capacities;
capacity bounds imply actual `Kernel.TraceSafe`. It adds no arbitrary runner or
preparation callback and does not change `AlgorithmLaunch`.

[UndefProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/UndefProgramIO.lean) adds finite typed
logical inputs for the actual fallback values of masked loads. Its binding
reads the initial state's `undef`; it neither initializes it nor restricts
unobserved fallback values. Only memory channels require backing capacity.
The fixed source must still execute successfully and establish typed outputs
and the exact raw frame. This preserves, for example, multinomial sampling's
original seed domain on padded rows without assuming zero fallback seeds.

[ReductionProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ReductionProgramIO.lean) composes
fixed source invocations with finite FP32 sum stages. Each sum has a closed
zero/input/add tree whose leaves cover every input exactly once. It reads the
actual predecessor snapshot, retains FP32 loads/additions/final casts, and
stores in the declared finite order. It reuses the selected library sum model;
it makes no claim about Torch's internal implementation. Successful outputs
and the raw frame are unconditional on capacity witnesses. The
`FPReductionProgramIO.lean` regressions check snapshot and output aliases,
source predecessors, exact storage tags, empty stages, bounds and source failure.

Semantic regression checks live in `bench/tests/FP*ProgramIO.lean`,
`FPStructuralSafety.lean`, `FPInputBindingIO.lean`, `FPAddressedInputsIO.lean`,
`FPObservedProgramEquivalence.lean`, `FPIntegerStageIO.lean`,
`FPIntegerOutputIO.lean`, `AlgorithmScheduledIO.lean` and `AlgorithmProgramIO.lean`, registered in the
`FPFrontendRegressions` target.
The added `PreparedAlgorithmProgramIO.lean`, `FPUndefProgramIO.lean` and
`FPMultinomialUndef.lean` checks cover private initialization, predecessor memory,
nonzero masked fallbacks, signed seed boundaries, and rejection of source
failure or unbacked active accesses.

[ObservedInputCellsIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/ObservedInputCellsIO.lean)
composes exact typed input-cell binding with a pure output observer. It retains
the existing source execution, access safety, output dtype and raw exterior
frame obligations. Even a constant observer cannot hide a wrong storage dtype
or a write outside the output windows. `FPObservedInputCellsIO.lean` checks
those cases and composition with an actual successful source implementation.

[BackendMemory](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/BackendMemory.lean) provides successful
whole-source memory equivalence under one explicit Structural backend. Both
executions must succeed for every initial state and agree on every raw memory
cell. It introduces no finite-window input contract; access safety belongs to
the separate IO application. Identical failures do not satisfy this relation.

[PerAddressProgramIO](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/PerAddressProgramIO.lean) supports
existing per-address atomic linearization proofs. Each event uses a successful
complete source invocation and its exact typed result cell. Every listed CTA
has a safety and success obligation, even with no output lanes. Independent
orders require exact occurrence accounting, source-output locality, and proof
that omitted events preserve the selected cell. This interface provides no
general concurrent GPU semantics; each consumer must connect it to its existing
source and linearization proof. Tests are in `FPPerAddressProgramIO.lean`.

The existing TritonBench [vector addition correctness theorem](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/tritonbench_g/vector_addition/Correct.lean)
uses the real `addIO … ⊨ fun xs ys i => xs i + ys i` surface. The separate
[correctness example](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/examples/TritonBenchVectorAddition/Correct.lean)
defines its fp32 kernel and real-valued IO specification:

```lean
specification vector_addition_correct (nElements blockSize : Nat) :
    Spec.Real (addIO nElements blockSize ⊨ fun xs ys i => xs i + ys i)
```

It reuses the existing TritonBench proof through the kernel's mathematical
projection. `real_projection` lives in this file; no numerical assumptions are needed.

The [floating-point companion](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/examples/TritonBenchVectorAddition/FPEquiv.lean)
compares that actual kernel with a variant changing only `output = x + y` to
`output = y + x`, with fp32 operations and symbolic element count and block size.
Its headline is:

```lean
open scoped VeriTile.Spec

specification vector_addition_equiv (nElements blockSize : Nat) (R : Rules blockSize) :
    originalKernel nElements blockSize ≡[R] optimizedKernel nElements blockSize
```

This reuses the existing equivalence notation. Here `R` is the rule model;
its table stores the experiment configuration and results, and its `add_comm`
field requires admission of the atom. Neither `experiment` nor `evidence` is
a public theorem argument. `#print_fp_assumptions vector_addition_equiv` lists
only the referenced atomic assumption names, such as `add_commute`. Under this scope, the notation
elaborates to `Spec.FloatingPoint`; historical KernelIO scopes still use their
original `RoundingModel` relation.

The FP file writes both Triton kernels directly and does not import the
correctness file. Its equivalence proof uses only its atomic assumptions.

Experimental parameters select the available assumptions. The subsequent proof
uses those assumptions without matching kernel sizes to experimental sizes or
checking input distributions. `Rules blockSize` only instantiates typed syntax;
it imposes no equality with the experiment's block size. The public theorem is
parameterized, so it does not need fixed `rows`, `columns` or launch constants.

The intended user workflow starts with a Python shape/input-sampling profile
(plus the numerical configuration), checks the predefined candidate rules with
two-gates on Triton/GPU, generates `R`, and automatically attempts a proof using
the existing proof agent and comparator with those rules frozen beforehand.
Changing the profile generates a new rule set and reruns dependent proofs;
failure to find a derivation is reported rather than filled by a new assumption.
The distribution defines the atom test environment. The theorem is conditional
on `R`, not a claim that arbitrary intermediate operands follow that distribution
or that the whole kernel passed the gates. This automation remains to be connected;
see the workflow in [FloatingPointPrimitives.md, section 7.1](https://github.com/VeriInfra-Lab/VeriTile/blob/main/documents/FloatingPointPrimitives.md#71-用户流程配置--检查原语--生成规则集--自动证明).

The table contains one `ADD-COMMUTE` atom: the actual differing assignment
fragments. Lean checks the syntax decomposition and lifts that atom through
unchanged loads and store. It does not assume the whole kernel result. The
original real correctness theorem remains separate and is not used to smuggle
real `add_comm` into a floating-point equivalence proof.

The implementation in [Spec.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Spec.lean) separates three things:

1. `AtomicRule` and `RuleEntry` name a numerical rewrite and its configuration,
   identity and evidence. Checks apply to atoms, not to the public theorem.
2. `AcceptedAtom` requires validated/replayed evidence, matching identity, an
   artifact and two allowed gate results. `AcceptedAssumptions` collects these
   obligations for a table. PASS labels alone cannot admit a rule.
3. `Derivation` supplies reflexivity, symmetry, transitivity and common sequential
   context. `FloatingPoint` additionally preserves the implementation signature.
   `ProgramSyntax` supplies the auditable signature/body view; ComputeKernel's
   view retains its input/output lists and actual ComputeStmt sequence.

Admitting an atom means **assuming its relation in this formal theory**. The
symmetry and composition rules are part of that policy; they do not claim that
the original directed statistical tests reverse or compose. There is no
conversion to Lean equality of concrete IEEE values, no global floating-point
ring instance, and no inferred whole-kernel statistical guarantee. Hypotheses
retain the experiment configuration as provenance for selecting the assumptions.
Rule matching preserves operation identity, precision and syntactic side conditions;
kernel shape and block size need not equal the experiment's shape or block size.
Statistical claims about an actual use site's operands need their own sampling
justification. The current context rule is syntactic sequence framing, not a
proof of distribution transport.

`#print_fp_assumptions name` walks the FP proof and instantiated helper proofs,
printing only the referenced `Derivation.atom` names in lowercase snake case.
Configuration and gate status remain in the rule records and are not printed.
It does not enumerate unused entries in the rule table or print the specification.
Repeated instances are deduplicated; distinct configurations stay separate.
Opaque FP proof premises are marked unresolved rather than assumed to use no atoms.
Reachable proof branches are included, so this is not a minimal logical dependency set.
The command accepts ordinary FP theorems as well as `specification` headlines.
Printing does not discharge a premise or perform an axiom audit. The example runs
the separate axiom audit silently on success. The older `#print_spec name full`
remains available for the internal configuration/evidence and dependency audit.

The calculus, reporting, example and rejection checks are implemented. The
[GPU experiment runner](https://github.com/VeriInfra-Lab/VeriTile/blob/main/experiments/floating_point/README.md) now supplies
14 fixed-size atomic expression pairs, Python configuration, gates and CPU result replay.
Composite algorithms, reductions, scans, dot rewrites and memory/layout properties
are excluded from admission; their equivalences require Lean derivations.
The report exporter freezes the current GPU report’s accepted instances
as Lean data, and the worked example binds fp32 ADD-COMMUTE to its concrete
assignment fragments. The model premise `R.add_comm` explicitly trusts that
external result and binding; Lean checks the derivation under it. Other atomic
syntax bindings remain separate work. See the [worked example](https://github.com/VeriInfra-Lab/VeriTile/blob/main/experiments/floating_point/EXAMPLE.md)
for generation, build and inspection commands.

Historical unwrapped or abstract cast/store-rounding headlines keep their old
meaning. They do not become atom-based FP equivalence merely by renaming them.
Exact FP facts remain internal helper lemmas, not a third public specification.

## Underlying library surfaces

This document explains which public theorem surface to use when proving
properties of `ComputeKernel`s. Exact surfaces live in
[`VeriTile.Triton.Correctness`](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Correctness.lean); rounding
surfaces live in [`Float.Refine`](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/Refine.lean).

## Rounding models and exact surfaces

The interfaces are defined in `VeriTile/Triton/Correctness.lean` (exact
semantics) and `VeriTile/Triton/Float/Refine.lean` (rounding models):

- `ComputeCorrect.Realizes_without_Rounding kernel s write expected` observes
  outputs under exact semantics, with `expected : ι → α`.
- `ComputeRefine.Realizes kernel s write expected` quantifies over all
  `RoundingModel`s, with `expected : RoundingModel → ι → α`.
- `ComputeRefine.Refines R lhs rhs s scratch` and `RefinesAt R ...` compare two
  kernels under a supplied model `R`. Their exact versions have the
  `_without_Rounding` suffix.
- `ComputeCorrect.Post`, `General`, and the `Output*` wrappers retain exact semantics.

A `RoundingModel` requires identity on the real channel (`round_real`) and
idempotence (`round_idem`). Monotonicity, oddness, and grid nesting are optional
hypotheses. The model records rounding at explicit float casts and stores;
it does not model IEEE-754 rounding at every arithmetic operation or establish
numerical error bounds. At `R := .triv`, `execR` degenerates to `exec`; the
bridge lemmas are described below.

## KernelIO `⊨` — the per-kernel headline surface

The surfaces below (`Realizes`, `Refines`, …) are the *library* vocabulary. The
surface that the bench corpus actually states its headlines on is the **KernelIO
`⊨` triple**, defined in
[`VeriTile/Triton/Memory/KernelSpec.lean`](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Memory/KernelSpec.lean). The signature variants cover different input/output arities, masks,
indices, and streaming windows. Consult the
[final contract review](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/validation/contracts.json) for
per-port coverage.

### What a `KernelIO` is

A `KernelIO*` record is a kernel's **IO signature**: which regions it reads and
writes, where program `pid`'s window starts in each of them (`read`/`write`),
which lanes are active (`readMask`/`writeMask`), the tile length `B`, and — on
the rounding skins — the output element type `outDType`. The many `structure`s
("skins") differ only in the *shape* of that signature: how many inputs and
outputs, how many program axes, whether metadata scalars or a gather index sit
between the pointer and the data.

### What `io ⊨ f` says

```lean
open scoped VeriTile.Triton.KernelIO₂ in
specification my_kernel_correctness … :
    myIO … ⊨ fun xs ys j => xs j + ys j
```

Unfolded, that is one Hoare triple over **flat pointer memory**: for every
disjoint flat placement of the declared buffers, every program id whose windows
are in bounds, and every launch state whose input windows hold `xs`/`ys`, the
*translated pointer kernel* terminates, every declared output cell holds
`f xs ys j`, and **every other flat cell is unchanged** (the frame).

That last conjunct is the part a hand-rolled `∃ sF, exec … = some sF ∧ …`
headline usually omits, which is why it is not merely a different spelling —
see the final contract review for the ports that
still owe it.

### The notations

| Notation | Meaning |
|---|---|
| `io ⊨ f` | exact-ℝ: `exec`, no rounding |
| `io ⊨[R] f` | rounding: `execR R`, output rounded at the signature's `outDType` |
| `io ⊨[R, dtype] f` | same, with `outDType` written out — use this whenever a file states more than one output channel, since the 3-hole form prints an `.fp16` face and a `.real` face identically |
| `io₁ ≡[R] io₂` | kernel-vs-kernel refinement, the `⊨` counterpart of `Refines` |

`io ⊨[R] f` uses the supplied model `R`; universal quantification must be
explicit in the theorem. A theorem valid for every `R` can be instantiated at
`.triv`. Boundary quantization has narrowing content when the output dtype is
narrow; the `.real` channel satisfies `R.round .real = id`.

### Proving one

Each skin ships an assembly lemma `<Skin>.Implements.intro` (and
`.ImplementsR.intro`) that reduces `⊨` to three per-kernel obligations and does
the flat-memory transport once:

1. `FlattenOk` — the kernel's ops are in the flat-memory bridge fragment;
2. `TraceSafe` — the per-execution safety walk, given the window-in-bounds contract;
3. `hrun` — the **region-model** Hoare triple: termination, output values, frame.

Only (3) is mathematical content; (1) and (2) are mechanical walks.

## Quick-Pick Table

| Goal | Use |
|---|---|
| **A bench-port / showcase headline** | a `KernelIO` skin's `⊨` (exact) or `⊨[R]` (rounded) — see the section above |
| … kernel-vs-kernel, on the same surface | `io₁ ≡[R] io₂` |
| One kernel realizes an output spec (rounded) | `ComputeRefine.Realizes` |
| … the exact-ℝ idealization of that spec | `ComputeCorrect.Realizes_without_Rounding` |
| One kernel refines another, writes-equality (rounded) | `ComputeRefine.Refines` |
| … the exact-ℝ idealization | `ComputeRefine.Refines_without_Rounding` |
| Two kernels agree pointwise per declared address | `ComputeRefine.RefinesAt` (exact: `RefinesAt_without_Rounding`) |
| Complete sequential source grid | `ProgramIO.Implements` or `UKernelIO.RegionalScheduledImplements` |
| Value/index pair on every lane | `ComputeCorrect.OutputPair` |
| Value/index pair on active lanes only | `ComputeCorrect.OutputPairWhere` |
| Two kernels produce equal value/index pairs | `ComputeRefine.OutputPairEq` / `OutputPairEqWhere` |
| Custom postcondition over the final state | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| Relation over arbitrary initial states (rare) | `ComputeCorrect.General` / `ComputeRefine.General` |

The lower-level `ComputeKernel.ComputeCorrect` and
`ComputeKernel.ComputeRefine` definitions remain the implementation layer.
New example theorem statements should normally not expose those names directly.

## Execution, termination, and frame

`Realizes_without_Rounding` and `ComputeRefine.Realizes` constrain outputs of
successful executions; they do not by themselves establish success or
termination. `Refines` likewise compares the results when both executions
succeed. `RealizesFrame_without_Rounding` additionally preserves memory outside
the write map. For a bench headline that bundles termination, outputs, and
frame, prefer an appropriate `KernelIO` `⊨` / `⊨[R]` signature and inspect its
specific preconditions.

## Output Write Maps

The most general output surface separates two maps over the same logical output
index:

```lean
abbrev ComputeCorrect.WriteMap (ι : Type) := ι → Option MemCellAddr

ComputeCorrect.Realizes_without_Rounding
  (kernel := k)
  (initialState := s)
  (write := write)
  (expected := expected)
-- post: ∀ i, match write i with
--   | some addr => read final addr = expected i
--   | none => True
```

`Realizes_without_Rounding` is overloaded by the expected value type. `ℝ` specs use
`BlockState.readMem`, `Nat` specs use `readMemValue .nat`, and `MemCell` specs
use exact algorithm-layer cell equality. Use it when the spec is naturally a
write-indexed written-cell contract or spans multiple regions. For common
scalar/tensor readback theorems, the ergonomic wrappers (`OutputScalar`,
`OutputArray`, `OutputNatScalar`) remain available.

Masked stores should usually be stated with `WriteMap.writeIf`:

```lean
write := ComputeCorrect.WriteMap.writeIf
  (fun i : Fin BLOCK_SIZE => base + i.val < N)
  (fun i => (out, base + i.val))
```

Then the theorem states what active lanes wrote. Preservation of non-written
addresses belongs to frame/preserve theorems.

The standard proof step is:

```lean
rw [ComputeCorrect.realizes_writeIf_iff]
```

which turns the output obligation into `∀ i, mask i → read final (addr i) =
expected i`.

Whole-grid public theorems use shared IO over the complete source schedule and
global logical output windows. Final-state execution and frame assembly stay in
supporting proofs. A per-program theorem does not replace a previously proved
complete grid.

### Grid launches today

`ProgramIO` supports finite Structural source schedules, including different
kernels and primitive backends per launch. `RegionalScheduledKernelIO` supports
finite schedules of one Algorithm source. Both thread actual predecessor
states and require successful execution and source-access safety. They do not
model arbitrary concurrent or per-address atomic orders; retain the existing
execution model when a task needs those orders.

The older `Kernel.ForAllProgramsSome` remains a per-program-local result: each
typed grid index runs from `s.withGridIndex idx`. It can serve as a supporting
lemma, but by itself it does not establish one complete sequential grid result.

## Single-Kernel Correctness

Single-kernel correctness checks one kernel against a mathematical or
algorithmic specification.

### Scalar outputs

```lean
ComputeCorrect.OutputScalar k s out offset expected
-- post: s'.readMem out offset = expected

ComputeCorrect.OutputNatScalar k s out offset expected
-- post: s'.readMemValue .nat out offset = expected
```

Use these for scalar reductions such as a final `max`, `sum`, or
`argmax`-style integer index.

### Tensor outputs

```lean
ComputeCorrect.OutputTile k s view expected
-- post: ∀ idx, TensorView.observe (some s') view idx = some (expected idx)

ComputeCorrect.OutputArray k s view expected   -- 1D specialization
-- post: same, with `expected : Fin n → ℝ`
```

`OutputArray` is the 1D `n`-shape specialization of `OutputTile` and is the
preferred surface when the spec is naturally a `Fin n → ℝ` function. Both are
thin wrappers over `Realizes_without_Rounding` through `WriteMap.ofTensorView`; they exist to
keep tensor-view theorem statements readable.

### Value/index pair outputs

```lean
ComputeCorrect.OutputPair k s valueRegion indexRegion offset
  expectedValue expectedIndex
-- post: every lane satisfies the value spec and the typed Nat index spec

ComputeCorrect.OutputPairWhere k s valueRegion indexRegion offset
  active expectedValue expectedIndex
-- post: same, but only on lanes where `active i` holds
```

Use these for kernels with paired outputs such as
`tl.max(..., return_indices=True)`. `OutputPairWhere` is the right choice
when masking restricts which lanes participate.

These remain dedicated definitions rather than `Realizes_without_Rounding` wrappers because the
two channels have different readback carriers (`ℝ` for value, `Nat` for index)
and the `Realizes_without_Rounding` typeclass dispatches on a single carrier per call. A future
revision could introduce a per-lane carrier typeclass to subsume them; not
worth doing until a second heterogeneous-output kernel appears.

## Kernel Refinement

### `Realizes` vs `Refines` — the naming scheme

VeriTile splits the two verification questions by name (each name is the
rounding surface; append `_without_Rounding` for the exact-ℝ idealization):

- **`Realizes`** — *a kernel realizes a spec* (one kernel vs expected
  outputs). `ComputeRefine.Realizes` is the rounding form;
  `ComputeCorrect.Realizes_without_Rounding` is the exact single-kernel
  workhorse that most ported kernels use.
- **`Refines`** — *a kernel refines another kernel* (two kernels compared to
  each other). `ComputeRefine.Refines` (writes-equality) and
  `ComputeRefine.RefinesAt` (pointwise per-address relation), with their exact
  mirrors `Refines_without_Rounding` / `RefinesAt_without_Rounding`.

### Writes-equality: `ComputeRefine.Refines`

The canonical two-kernel surface is `ComputeRefine.Refines`. Under a rounding
model `R` it runs both kernels from the same initial state through `execR R`
and asserts that they performed **the same writes** — the two final memories
agree at every cell outside a declared list of `scratch` regions:

```lean
ComputeRefine.Refines R lhs rhs s scratch
-- := ExecRefineR R lhs rhs s (fun l r =>
--      ∀ region ∉ scratch, ∀ offset, l.mem region offset = r.mem region offset)
```

Same write locations, same written values, one equation — for the given `R`.
`scratch` names the regions the two kernels are *allowed* to disagree on — a
pipeline's intermediate tensors (e.g. `[S]` for a fused-vs-unfused SwiGLU pair,
or the `zReg`/`siluReg` temporaries in `FusedSiLU`). Pass `[]` when the two
kernels must agree on all of memory. This is the surface every
`*_refinement_view` theorem in `bench/examples/` lands on; the exact-ℝ variant
`ComputeRefine.Refines_without_Rounding` (no `R`, running under `exec`) is the
idealization used by `bench/examples/FusedSiLU/Proofs/LegacyRealEquiv.lean`.

### Pointwise: `ComputeRefine.RefinesAt`

When the two kernels write to **different** target cells, or the comparison is
a non-equality relation, use the pointwise form. It relates the two kernels'
outputs through two independent `WriteMap`s:

```lean
ComputeRefine.RefinesAt R lhs rhs s lhsWrite rhsWrite relation
-- post: ∀ i, match lhsWrite i, rhsWrite i with
--   | some la, some ra => relation i (read lhs' la) (read rhs' ra)
--   | _, _ => True
```

The carrier types of the two reads are inferred independently, so `RefinesAt`
covers heterogeneous-layout comparisons and proof middleware. Use the same
write map on both sides for ordinary same-buffer equivalence. `Refines` is the
whole-memory form to reach for first; `RefinesAt` is the escape hatch when you
genuinely need per-side values.

### Rounding-model surfaces (narrow float, #1)

The unqualified surfaces are the rounding surfaces: each is parametric over a
`R : RoundingModel` (`round : FloatDType → ℝ → ℝ`, with idempotence a defining
field) and executes under the R-threaded semantics `execR`. They live in
[`VeriTile.Triton.Float.Refine`](https://github.com/VeriInfra-Lab/VeriTile/blob/main/VeriTile/Triton/Float/Refine.lean):

- `ComputeRefine.Realizes kernel s write expected` — single kernel vs an
  `R`-annotated spec `expected : RoundingModel → ι → α`. The spec's shape
  *is* the rounding-event ledger: no `R.round` means the observed path is
  rounding-free; one `R.round` means one final quantization; nested
  `R.round`s count one event each.
- `ComputeRefine.Refines R lhs rhs s scratch` — the writes-equality pair
  surface under `R`.
- `ComputeRefine.RefinesAt R lhs rhs s lhsWrite rhsWrite relation` — the
  pointwise pair surface under `R`.

The exact-ℝ idealizations are the `*_without_Rounding` mirrors, and they
**degenerate out of** the rounding surfaces at the trivial model `.triv` (where
`execR` collapses onto `exec`): the bridge `ComputeRefine.Realizes.toRealizes_without_Rounding`
("rounding claim implies ideal correctness") turns any `Realizes` into an
ordinary `ComputeCorrect.Realizes_without_Rounding` for `expected .triv`.
Degeneration lemmas `refines_triv_iff` / `refinesAt_triv_iff` recover
`Refines_without_Rounding` / `RefinesAt_without_Rounding` the same way. The
gold-standard walkthrough for the ∀R compositional pattern is
[`bench/examples/FusedSwiglu/Proofs/LegacyRealEquiv.lean`](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/examples/FusedSwiglu/Proofs/LegacyRealEquiv.lean); the
boundary-rounding showcases (LogSumExp, softmax, Welford, LayerNorm, FusedSiLU)
land on `Refines R` for a fixed `R`.

### Equality helpers

For ordinary optimization proofs that read back the same buffer on both sides,
the equality helpers below are thin wrappers over `ExecRefine` and remain the
most readable choice.

### Scalar equality

```lean
ComputeRefine.OutputScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMem lhsOut lhsOffset = rhs'.readMem rhsOut rhsOffset

ComputeRefine.OutputNatScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMemValue .nat ... = rhs'.readMemValue .nat ...
```

### Tensor equality

```lean
ComputeRefine.OutputTileEq lhs rhs s lhsView rhsView
-- post: ∀ idx,
--   TensorView.observe (some lhs') lhsView idx
--     = TensorView.observe (some rhs') rhsView idx

ComputeRefine.OutputArrayEq lhs rhs s lhsView rhsView   -- 1D specialization
```

### Value/index pair equality

```lean
ComputeRefine.OutputPairEq lhs rhs s
  lhsValueRegion lhsIndexRegion rhsValueRegion rhsIndexRegion offset

ComputeRefine.OutputPairEqWhere lhs rhs s ... offset active
```

Use the `Where` variant when the equality only needs to hold on active lanes.

## Escape Hatches

`Post` fixes the initial state but accepts any postcondition over the
final state:

```lean
ComputeCorrect.Post k s (fun s' => ...)
ComputeRefine.Post lhs rhs s (fun lhs' rhs' => ...)
```

`General` quantifies over the initial state itself, taking a relation
over `(s0, s')`:

```lean
ComputeCorrect.General k (fun s0 s' => ...)
ComputeRefine.General lhs rhs (fun s0 lhs' rhs' => ...)
```

`General` should be reserved for theorems that genuinely need to be
parametric in the initial state (rare). Most proofs fix `s` and use
`Post`, or pick an `Output*` helper.

## Gap Policy

The `ComputeCorrect.*` / `ComputeRefine.*` helpers use the default
`GapPolicy.ignore`, matching current Real-first algorithm proofs. If a
theorem needs to record an externally checked compute-to-algorithm gap,
state the underlying `ComputeKernel.ComputeCorrect` /
`ComputeKernel.ComputeRefine` directly with:

```lean
gap := .require contract
```

or add a dedicated helper for that theorem family. See
[`EraseDType.md`](/VeriTile/architecture/erase-dtype/) for the compute-to-algorithm projection
policy.

## Naming

Recommended theorem names:

- `<kernel>_compute_correct` for ordinary single-kernel correctness.
- `<kernel>_correct_view` for public manifest-compatible view surfaces.
- `<rewrite>_refinement_view` for public two-kernel refinement surfaces.

Execution-only helper lemmas may use `_exec_view` and direct `exec`
equalities. Selected benchmark public entries compose those helpers through
the shared IO or implementation-equivalence judgments described above;
`ComputeCorrect.*` and `ComputeRefine.*` remain supporting library surfaces.
Naming details:
[`TheoremSurfaces.md`](/VeriTile/proofs/theorem-surfaces/).
