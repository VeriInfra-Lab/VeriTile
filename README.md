# VeriTile

📖 **Docs site:** [veriinfra-lab.github.io/VeriTile/](https://veriinfra-lab.github.io/VeriTile/) (bench cookbook, status, architecture). Run locally: `./site/scripts/dev.sh`.

VeriTile embeds a typed Triton-style kernel DSL in Lean 4 and proves
correctness or refinement of those kernels against mathematical
specifications or against each other. The implementation embeds the kernel
language with `triton { ... }` syntax, defines an operational semantics over
typed `Op : TileDType → TileShape → Type` terms, and exposes
`ComputeCorrect` / `ComputeRefine` theorem surfaces for end users.

Algorithmic proofs run over the erased `.real` (mathematical) channel; the
optional `GapPolicy` records — but does not internally prove — the
compute-to-algorithm gap (IEEE-754 / PTX / TMA / concurrency), which stays
externally checked. See [Triton subset and gaps](documents/TritonSubset.md).

## What VeriTile Does

- **DSL**: typed Triton subset with `triton { ... }` macro, ND tile shapes,
  reductions, masks (`mask=`/`other=`), block-pointer ops, and bare
  `if`/`for` control flow.
- **Theorem surfaces**: `ComputeCorrect.Realizes_without_Rounding` for *one kernel vs a math
  specification*, `ComputeRefine.Refines` for *one kernel refining another*
  (writes-equality: the two final memories agree at every cell outside the
  declared scratch regions). Both project through `toAlgorithm?` and run on
  `Kernel.Correct_without_Rounding` / `Kernel.Refine` underneath.
- **Public specifications**: correctness uses real semantics (`Spec.Real claim`);
  implementation equivalence uses `lhs ≡[R] rhs` (`open scoped VeriTile.Spec`).
  Two-gates admits atomic rewrite assumptions; Lean checks their composition.
  `R` bundles the rule table and admissions; experiment bookkeeping stays there.
  `#print_fp_assumptions name` lists only the atomic assumption names referenced
  by an FP proof, such as `add_commute`. The GPU runner and trusted-report
  example are available, with separate [correctness](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/bench/examples/TritonBenchVectorAddition/RealCorrect.lean)
  and [FP equivalence](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/bench/examples/TritonBenchVectorAddition/FPEquiv.lean) specifications.
  See [specification meanings](documents/CorrectnessSurfaces.md#two-public-specification-meanings).
- **Narrow-float / rounding-model layer** (#1): an abstract `RoundingModel`
  (`round : FloatDType → ℝ → ℝ`, fields `round_real` (real-channel identity) and `round_idem` (idempotence)) threads a
  black-box rounding function through the semantics (`evalOpR` / `stepStmtR` /
  `execR`). The unqualified surfaces are the rounding-parametric ones —
  `ComputeRefine.Realizes` (single kernel vs an R-annotated spec) and
  `Refines` / `RefinesAt` (two kernels) run under a `RoundingModel R`; the
  exact-ℝ idealization is the qualified `*_without_Rounding` name, which the
  bridge `Realizes.toRealizes_without_Rounding` degenerates out to (as
  `ComputeCorrect.Realizes_without_Rounding`) at the trivial model. See the
  fused-vs-unfused SwiGLU showcase
  [`bench/examples/FusedSwiglu/Proofs/LegacyRealEquiv.lean`](bench/examples/FusedSwiglu/Proofs/LegacyRealEquiv.lean).
- **Examples**: 183 current TritonBench-G task directories, with proof
  contracts for their selected repaired implementations. The [historical contract review](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/validation/archive/contracts.json) preserves the earlier mathematical targets, implementation pairs, conditions and proof scope. See the
  [current source inventory](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/current_source_acceptance.md) for checked
  inputs and reproduction commands. Additional examples include FlashAttention-1
  forward, online softmax, Welford, LayerNorm, log-sum-exp.
- **CI gates**: `.github/workflows/bench-audit.yml` runs the full library build,
  source correspondence inventories, final contract/source checks, and library
  comparator on PRs and main pushes. The complete retained corpus and all 334
  benchmark interfaces are built and audited daily at 04:00 Asia/Singapore
  (20:00 UTC), or on manual dispatch. The daily corpus audit uses 64 shards
  through `scripts/bench/audit_trust.sh`, retaining elaboration, Lean trust/spec checks
  and official comparator replay. Its final gate requires every shard to pass.
  Run the complete applicable proof checks locally before merging proof changes.
  `.github/workflows/artifact.yml` runs `lake build` +
  `scripts/check-artifact.sh` (no `sorry`, axiom whitelist, manifest schema,
  doc-drift checks). `.github/workflows/site.yml` builds and deploys the docs
  site to GitHub Pages on every push touching `site/`.

Pending for complete kernel verification: concrete floating-point execution
(the bf16/fp32 scalar foundation is implemented), PTX-level codegen,
detailed concurrency (atomics / async-copy serialization, beyond the
projection boundary), Python wrapper execution.

## Quick Start

### Run a checked example

Install Git and [Lean's toolchain manager (`elan`)](https://lean-lang.org/install/),
then run:

```bash
git clone https://github.com/VeriInfra-Lab/VeriTile.git
cd VeriTile
lake exe cache get
lake build bench.examples.VectorAdd.RealCorrect bench.examples.VectorAdd.FPEquiv
lake env lean bench/examples/VectorAdd/RealCorrect.lean
```

The repository pins the Lean and Mathlib versions. The cache command downloads
precompiled dependencies; `lake build` builds the example and its dependencies.
The last command compiles the complete vector-add example and checks its
proofs in Lean. For the additional official comparator audit, follow the
[verification setup](scripts/README.md#setup). An agent CLI is optional.

The steps below show the vector-add body and its public real-correctness
specification from the complete example.

### 1. Write a `ComputeKernel`

Kernels are region-polymorphic: memory regions arrive as `RegionName`
parameters; `tl.load` / `tl.store` use first-class pointer expressions.

```lean
import VeriTile.Triton.DSL.CallSignature
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("x_ptr", .fp32), ("y_ptr", .fp32), ("out_ptr", .fp32)]
  scalars := []
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr", "y_ptr"]
  writePorts := ["out_ptr"]

def addKernel (x_ptr y_ptr out_ptr : RegionName) (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [] in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(out_ptr + offsets, output)
  }
```

The body copies the registered [Python function](bench/examples/VectorAdd/VectorAdd.py).
Pointer types and constexpr parameters stay outside it. Reusable bodies use
`def nameSource : SourceProcedure := with_triton_params ... in triton { ... }`;
see the [source-authoring standard](AGENTS.md#kernel-源码与文件职责).

### 2. Choose a theorem surface

For new public headlines, express correctness against the mathematical formula
in real semantics (`Spec.Real`); use `lhs ≡[R] rhs` in the `VeriTile.Spec` scope
for implementation equivalence derived from two-gates-admitted atomic rules. Both keep the `specification` keyword.
After importing `VeriTile.Meta.StatementAudit`, `#print_fp_assumptions name`
reports the experimental assumptions and strict model certificates used by the proof.

| Goal | Use |
|---|---|
| Kernel matches an independent mathematical formula | `Spec.Real (io ⊨ fun ... => formula)` |
| Original and candidate implementations are equivalent under rules `R` | `original ≡[R] candidate` |

Full surface guide: [CorrectnessSurfaces.md](documents/CorrectnessSurfaces.md).

### 3. State and prove the public specification

The example's `IO.lean` binds the input/output windows. Its `Proofs/Real.lean`
provides execution, safety and frame lemmas used by `RealCorrect.lean`:

```lean
import bench.examples.VectorAdd.IO
import bench.examples.VectorAdd.Proofs.Real
open VeriTile Triton
open VeriTile.Bench.Examples.VectorAdd
open scoped VeriTile.Triton.KernelIO₂

specification add_kernel_correctness (B : Nat) (hB : 0 < B) :
    Spec.Real (addIO B ⊨ fun xs ys i => xs i + ys i) := by
  refine KernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact addKernel_flattenOk "x" "y" "out" B
  · intro bounds s h1 h2 h3
    exact addKernel_traceSafe "x" "y" "out" B bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    exact addKernel_region_run B hB s₀ xs ys hx hy
```

The standard pattern: `ComputeKernel.computeCorrect_of_toAlgKernel rfl`
discharges the projection, then `simp` reduces `exec` to the body
recurrence; the algebraic content closes by `simp` on the spec or by
invoking a math identity from `Mathlib`. For equivalence proofs, `split_equiv`
isolates changed statements and `solve_fp` applies checked rules and discharges
their premises. See [the tactic guide](documents/EquivalenceTactics.md) for
options and guarded examples. The LLM proof wrapper
`scripts/prove.sh` automates this loop and uses the official comparator to judge
the targets selected with `--theorem` against the original task.
See [setup and usage](scripts/README.md).
The artifact and bench check scripts also require comparator proof replay;
see [the shared verification gate](scripts/README.md#shared-comparator-gate).

### 4. Register in the kernel manifest

Add a row to [`scripts/kernel-manifest.tsv`](scripts/kernel-manifest.tsv)
so `scripts/check-artifact.sh` recognizes the theorem in CI. Schema and
naming conventions: [KernelManifest.md](documents/KernelManifest.md),
[TheoremSurfaces.md](documents/TheoremSurfaces.md).

See the [worked-example index](bench/examples/README.md) for the shared-source directory layout and proof scopes.

## Minimal Example

Elementwise vector add against the `addSpec xs ys i = xs i + ys i` math
spec — see [`bench/examples/VectorAdd/RealCorrect.lean`](bench/examples/VectorAdd/RealCorrect.lean).

## Floating-Point Equivalence Example

Vector addition with swapped operands — see
[`VectorAdd/FPEquiv.lean`](bench/examples/VectorAdd/FPEquiv.lean), alongside
[`VectorAdd/RealCorrect.lean`](bench/examples/VectorAdd/RealCorrect.lean). The FP proof
uses the experiment-selected `add_commute` atom; the real proof states the
pointwise mathematical formula. `#print_fp_assumptions add_kernel_equiv`
prints the atom used by the FP proof.

The [floating-point experiment workflow](experiments/floating_point/README.md)
provides configurable Triton/GPU candidate checks and CPU replay of two-gates
results. The trusted GPU report supplies 30 accepted atomic instances for
`lhs ≡[R] rhs`. Other worked examples are being migrated from their legacy
models; the [coverage ledger](documents/ExampleSpecificationMigration.md)
records completed cases and remaining proof requirements.

## Documentation Map

Task-oriented:

| Question | Doc |
|---|---|
| Which Triton constructs are supported? | [TritonSubset.md](documents/TritonSubset.md) |
| What semantic caveats affect theorem interpretation? | [SemanticCaveats.md](documents/SemanticCaveats.md) |
| Where does my new lemma / definition belong? | [CodeOrganization.md](documents/CodeOrganization.md) |
| Tactic conventions (incl. `erw` carrier-bridge) | [ProofConventions.md](documents/ProofConventions.md) |
| Which theorem surface should I use? | [CorrectnessSurfaces.md](documents/CorrectnessSurfaces.md) |
| How does dtype erasure work? | [EraseDType.md](documents/EraseDType.md) |
| How does memory safety / framing work? | [MemorySafety.md](documents/MemorySafety.md) |
| What's the GPU memory model? | [GpuMemoryModel.md](documents/GpuMemoryModel.md) |
| How are atomics / async copies modeled? | [ConcurrencySemantics.md](documents/ConcurrencySemantics.md) |
| How does the kernel manifest work? | [KernelManifest.md](documents/KernelManifest.md) |
| Naming conventions for theorem surfaces | [TheoremSurfaces.md](documents/TheoremSurfaces.md) |
| LLM proof wrapper | [scripts/README.md](scripts/README.md) |

## Repository Layout

[VeriTileBench](https://github.com/VeriInfra-Lab/VeriTileBench) hosts TritonBench-G, the
bug-detection and pretraining challenges, and their per-task validation records.
It builds against a locked VeriTile revision.

```text
VeriTile/
  Triton.lean              Umbrella prelude (`import VeriTile.Triton`)
  Triton/
    Core/                  AST (Kernel/ComputeKernel type, ComputeOp bits)
    Semantics/             Typed operational semantics (exec, execR)
    Memory/                BlockState, tensor views, readback
    DSL/                   `triton { ... }` front-end
    Math/                  Pure `(Fin N → ℝ) → ...` operators (+ Math/Erf)
    KernelLemmas/          Reusable bench-proof helpers (was Triton/Kernel/)
    Correctness.lean       Top-level correctness/refinement surfaces
                           (Kernel.Correct_without_Rounding, ComputeCorrect.*, ComputeRefine.*)
    Float/                 Floating-dtype machinery: dtype erasure +
                           the rounding model (RoundingModel, execR, Refine,
                           Pipeline)
    Launch/                Grid-launch composition / write footprints
    Concurrency/           Grid-wide atomic-add correctness (above Launch)
  Examples/                Worked correctness/refinement proofs
bench/examples/            Worked examples with source, Real and FP entry points
tests/lean/               Lean regressions
documents/                 Design notes, subset spec, surface guide
documents/archive/         Historical design and implementation plans
scripts/                   CI gate, kernel manifest, LLM proof wrapper
  bench/                   Benchmark inventories, source checks and audits
site/                      Documentation website and generated reference pages
verso/                     Slide deck / overview
```

## Verification

- `python3 scripts/check_docs.py` — tracked Markdown links and reference-copy freshness
- `lake build` — build the default `VeriTile` library target; standalone
  benchmarks/showcases and the GeLU/trust-report target are separate
- `lake build VeriTile VeriTileFull` — also build the full analysis and library trust report
- `lake build TritonBenchSpecExamples` — build all worked examples, including the independent real/FP pairs
- `bash scripts/check-core-examples.sh` — check Python/Triton source correspondence and build
  `RealCorrect` and `FPEquiv` for `VectorAdd` (addition), `RowWiseSum` (reduction order),
  and `LogExp` (guarded numerical rewriting)
- `scripts/check-artifact.sh` — `lake build` ∧ `no sorry` ∧ axiom
  whitelist ∧ kernel-manifest schema ∧ README/doc-term drift
- The VeriTile workflow runs the library and three Real/FP examples for each pull request.
  It runs the full library and comparator audit nightly and on manual dispatch.
  Run the separate benchmark audit with the release candidate pinned before tagging.

## Environment

- Lean 4 (`v4.29.0`) + Mathlib
- Optional, for agent-assisted proof automation: [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) +
  [`lean4-skills`](https://github.com/lean4-skills/lean4-skills)
- For artifact/bench verification and proof automation: Python 3 and the official comparator, lean4export,
  and landrun on Linux with a systemd user service; see
  [installation instructions](scripts/README.md#setup).

## Roadmap

Long-running project. Goal: bring real Triton kernels (forward, backward,
concurrency, production layouts / autograd) into Lean's proof scope with
minimal modification. Live roadmap:
[#1](https://github.com/VeriInfra-Lab/VeriTile/issues/1). Architecture and
decision log: [PLAN.md](PLAN.md).

## Team

VeriTile is a collaboration between Zenan Li, Kaiyu Yang, Ziran Yang,
Zhaoyu Li, Mike He, and Aarti Gupta.

## License

[MIT](LICENSE) © 2026 Zenan Li.
