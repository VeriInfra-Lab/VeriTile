import bench.tritonbench_g.multinomial_sampling.Proofs.FPWrapper
import bench.tritonbench_g.multinomial_sampling.Proofs.FPSoftware

namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.HostCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Complete logical grid of the actual source for all four score storage
profiles. Every batch row, partial final block, and untouched memory cell is
accounted for; the reference depends only on launch-initial memory. -/
specification multinomial_grid_backend_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (d : Storage) (p : Parameters) (initial : State α)
    (fresh : Host.FreshOutput p)
    (seeds : ∀ k, k < Host.grid p → SeedDomain p (Host.programState p initial initial k)) :
    ∃ final, Host.run rest P d p initial = some final ∧
      final.mem = Host.expectedMemory P d p initial :=
  Host.launch_correct rest P d p initial fresh seeds

/-- The same complete grid with binary16/bfloat16/binary32/binary64 storage,
concrete Philox and software fp32 operations. Native numerical refinement is
still an explicit deferred obligation. -/
specification multinomial_grid_software_correctness
    (rest : Algebra Word) (d : Storage) (plans : Plans) (p : Parameters)
    (initial : State Word) (fresh : Host.FreshOutput p)
    (seeds : ∀ k, k < Host.grid p → SeedDomain p (Host.programState p initial initial k)) :
    ∃ final, Host.run rest (software d plans) d p initial = some final ∧
      final.mem = Host.expectedMemory (software d plans) d p initial :=
  Host.launch_correct rest (software d plans) d p initial fresh seeds

/-- Both host result shapes, all shape-check branches, default arange/expand,
clone initialization, fixed host blocks, and complete grid execution in the
signed64-index profile. The early return needs neither valid index shape nor
seed values. Other supplied-index dtypes are outside this declaration. -/
specification multinomial_int64_indices_host_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (d : Storage) (p : Parameters) (a : Host.Arguments) (initial : State α)
    (score : a.scoreRank = 2) (nonempty : 0 < p.tokens)
    (index : p.tokens ≠ 1 → a.indexShape.any (fun sh => sh != [p.batches, p.tokens]) = Bool.false)
    (fresh : Host.FreshOutput p)
    (freshDefault : a.indexShape = none →
      p.indices ≠ p.scores ∧ p.indices ≠ p.seeds ∧ p.indices ≠ p.offsets)
    (seeds : p.tokens ≠ 1 → ∀ k, k < Host.grid (Host.wrapperParameters p a) →
      SeedDomain (Host.wrapperParameters p a)
        (Host.programState (Host.wrapperParameters p a)
          (Host.kernelInitial p a initial) (Host.kernelInitial p a initial) k)) :
    ∃ result, Host.invoke rest P d p a initial = some result ∧
      result.state.mem = Host.wrapperExpectedMemory P d p a initial ∧
      result.shape = if p.tokens = 1 then [p.batches, 1] else [p.batches] := by
  have _ := freshDefault
  exact Host.wrapper_correct rest P d p a initial score nonempty index fresh seeds

/-- Token-one returns signed64 zeros with shape B-by-1 before examining the
index argument; no floating or random-number assumption is needed. -/
specification multinomial_single_token_host_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (d : Storage) (p : Parameters) (a : Host.Arguments) (initial : State α)
    (shape : a.scoreRank = 2) (token : p.tokens = 1) :
    Host.invoke rest P d p a initial = some ⟨Host.zeros p initial, [p.batches, 1]⟩ ∧
    (∀ o, o < p.batches → (Host.zeros p initial).mem p.output o = .mk .int 0) ∧
    (∀ r o, (r ≠ p.output ∨ p.batches ≤ o) →
      (Host.zeros p initial).mem r o = initial.mem r o) :=
  Host.single_token rest P d p a initial shape token

#axiomsClean multinomial_grid_backend_correctness
#axiomsClean multinomial_grid_software_correctness
#axiomsClean multinomial_int64_indices_host_correctness
#axiomsClean multinomial_single_token_host_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MultinomialSampling.HostCorrect
