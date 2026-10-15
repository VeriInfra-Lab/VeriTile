import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchLoop
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchAssembly
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
local instance branchCompleteWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete source execution with all dynamic loops and all declared branch
choices. The output equation is an independent scalar/tree/memory recurrence;
no final result, intermediate result or numerical rewrite is a premise. -/
theorem correctness (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial : State Word)
    (hBN : 0 < p.blockRows) :
    ∃ final, FP.Structural.exec (engine rest cfg plans) (source profile out p) initial = some final ∧
      final.mem = expectedMemory cfg plans profile out p initial ∧
      (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o) := by
  let hd : State Word := (headerExecution rest cfg plans profile out p initial).val
  have hh : run (engine rest cfg plans) (header profile out p) initial = some hd :=
    (headerExecution rest cfg plans profile out p initial).property
  have hg : evalOp (engine rest cfg plans) none (guard profile out p) hd =
      some (fun _ => decide (selected p initial ≠ -1)) :=
    guardEval rest cfg plans profile out p initial
  by_cases hs : selected p initial = -1
  · refine ⟨hd, ?_, ?_, ?_⟩
    · change run (engine rest cfg plans) (source profile out p).surfaceBody initial = some hd
      rw [sourceShape, run_append, hh]
      simp only [Option.bind_some, run, step, hg]
      simp [hs]
    · simp only [expectedMemory, hs, ↓reduceIte]
      exact headerMemory rest cfg plans profile out p initial
    · intro r o _
      exact congrFun (congrFun (headerMemory rest cfg plans profile out p initial) r) o
  · let prepared : State Word := (setupExecution rest cfg plans profile out p initial).val
    have hp : run (engine rest cfg plans) (setup profile out p) hd = some prepared :=
      (setupExecution rest cfg plans profile out p initial).property
    have hi : Invariant cfg profile p initial prepared :=
      setupInvariant rest cfg plans profile out p initial
    obtain ⟨final, hr, hm⟩ := loopExecution rest cfg plans profile out p initial hBN
      (length p) 0 prepared (by omega) hi
    refine ⟨final, ?_, ?_, ?_⟩
    · change run (engine rest cfg plans) (source profile out p).surfaceBody initial = some final
      rw [sourceShape, run_append, hh]
      simp only [Option.bind_some, run, step, hg]
      simp only [hs, ne_eq, not_false_eq_true, decide_true]
      rw [runActive, run_append, hp]
      simp only [Option.bind_some, run, loopStmt, step, evalOp_unfold,
        hi.size, Option.bind_some]
      simp [bind, Option.bind, hr]
    · simp only [expectedMemory, hs, ↓reduceIte]
      rw [hm, setupMemory]
    · intro r o ho
      rw [hm, setupMemory]
      exact reference_frame cfg plans profile out p initial _ _ _ r o ho

/-- Every unused fallback algebra yields the same complete final memory. -/
theorem backend_independent
    (rest₁ rest₂ : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial : State Word)
    (hBN : 0 < p.blockRows) :
    ∃ final₁ final₂,
      FP.Structural.exec (engine rest₁ cfg plans) (source profile out p) initial = some final₁ ∧
      FP.Structural.exec (engine rest₂ cfg plans) (source profile out p) initial = some final₂ ∧
      final₁.mem = final₂.mem := by
  obtain ⟨final₁, h₁, hm₁, _⟩ := correctness rest₁ cfg plans profile out p initial hBN
  obtain ⟨final₂, h₂, hm₂, _⟩ := correctness rest₂ cfg plans profile out p initial hBN
  exact ⟨final₁, final₂, h₁, h₂, hm₁.trans hm₂.symm⟩

#axiomsClean correctness
#axiomsClean backend_independent
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
