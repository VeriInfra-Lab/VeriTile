# VeriTile Reference Documentation

Long-form reference for VeriTile's embedded Triton subset, semantic model,
and trusted-boundary policies. Each active doc is a *contract*: what is
modeled, what is not, where the boundary sits.

Closed-phase design notes and resolved research-problem records live in
[`archive/`](archive) — kept as historical artifacts; not load-bearing
for current work.

## Active Reference Docs

Task-oriented index (mirrors the top-level [`README.md`](../README.md)
Documentation Map):

| Question | Doc |
|---|---|
| Which Triton constructs are supported? | [TritonSubset.md](TritonSubset.md) |
| What semantic caveats affect theorem interpretation? | [SemanticCaveats.md](SemanticCaveats.md) |
| Where does my new lemma / definition belong? | [CodeOrganization.md](CodeOrganization.md) |
| Tactic conventions (incl. `erw` carrier-bridge) | [ProofConventions.md](ProofConventions.md) |
| Which theorem surface should I use? | [CorrectnessSurfaces.md](CorrectnessSurfaces.md) |
| How do I distinguish real/FP specifications and print their assumptions? | [Two public specification meanings](CorrectnessSurfaces.md#public-specification-meanings), [TrustAudit.md](TrustAudit.md) |
| Naming conventions for theorem surfaces | [TheoremSurfaces.md](TheoremSurfaces.md) |
| How does the kernel manifest work? | [KernelManifest.md](KernelManifest.md) |
| How is the axiom-clean / sorry-free trust audit run? | [TrustAudit.md](TrustAudit.md) |
| How does dtype erasure work? | [EraseDType.md](EraseDType.md) |
| How does memory safety / framing work? | [MemorySafety.md](MemorySafety.md) |
| What's the GPU memory model? | [GpuMemoryModel.md](GpuMemoryModel.md) |
| What justifies physical load/store alignment, masks and Torch bindings? | [Physical byte-address contracts](GpuMemoryModel.md#physical-byte-address-contracts) |
| How are Tensor owning devices connected to CUDA launches? | [LaunchDeviceAudit.md](LaunchDeviceAudit.md) |
| How are atomics / async copies modeled? | [ConcurrencySemantics.md](ConcurrencySemantics.md) |
| ApproxGeLU midrange certified-error strategy | [ApproxGeluPhiStrategy.md](ApproxGeluPhiStrategy.md) |

## Architecture / Roadmap

The end-to-end project plan and roadmap live in the repo root:

- [`../PLAN.md`](../PLAN.md) — architecture, decision log, phase status
- Live roadmap: GitHub issue
  [`#1`](https://github.com/VeriInfra-Lab/VeriTile/issues/1)

## Paper Preparation

- [MLSys 2027 submission plan](MLSys2027Plan.md) — paper positioning,
  method claims, proof evidence, evaluation requirements and reproducibility
  (in Chinese). This planning document is separate from the semantic contracts
  and the long-term project roadmap.
- [Two-gates floating-point acceptance contract](TwoGatesAcceptance.md) —
  fixed input budgets, weighted means, restored-distribution peaks, decisions
  and evidence boundaries (in Chinese).
- [Floating-point execution primitives and proof interfaces](FloatingPointPrimitives.md) —
  typed source execution, strict software certificates, experimental atoms,
  Lean rule dependencies and the numerical evidence workflow (in Chinese).
- [Floating-point rewrite rules and acceptance table](FloatingPointRewriteRules.md) —
  candidate transformations, kernel-checked scalar witnesses, and per-configuration
  gate result fields, with strict certificates and experimental evidence classified separately (in Chinese).
- [Run floating-point experiments](../experiments/floating_point/README.md) —
  Python configuration, GPU commands, resume, result packaging and CPU replay.
- [Example specification migration](ExampleSpecificationMigration.md) —
  case-by-case real-correctness and FP-equivalence coverage, checked pairs,
  and missing atomic or structural proof requirements.

## Archive

Closed-phase notes are preserved in [`archive/`](archive):

- `DslMacroOptions.md` — macro design exploration, decision: option C
  (`triton { ... }` block macro). Closed.
- `ForLoopInvDesign.md` — Phase B `forLoop_inv` interface decisions.
  Closed; current implementation matches.
- `ResearchProblemPointerRegion.md` — RP1: pointers vs named regions.
  Resolved: named regions throughout.
- `ResearchProblemAddressTyping.md` — RP2: ℝ-uniform vs Nat-bifurcated
  `Value`. Resolved: bifurcated `Value`.
- `Proposal.md` — initial project proposal. The
  technical content has evolved beyond it; kept for historical context.
- `plans/` and `specs/` — dated implementation plans and designs moved from
  the former `docs/superpowers/` tree; their paths and completion notes describe
  the source revisions at the time they were written.
