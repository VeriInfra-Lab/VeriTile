import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBodyValue
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Fuel bounds the number of positive-stride output blocks; no result or
numerical equality is assumed. Every block sees the prior block's memory. -/
theorem loopExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial : State Word)
    (hBN : 0 < p.blockRows) (fuel n : Nat) (current : State Word)
    (hbound : length p-n ≤ fuel) (h : Invariant cfg profile p initial current) :
    ∃ final, FP.Structural.range (engine rest cfg plans) "n" n (length p) p.blockRows
      (body profile out p) current = some final ∧
      final.mem = reference cfg plans profile out p initial fuel n current.mem := by
  induction fuel generalizing n current with
  | zero =>
    have hn : ¬ n < length p := by omega
    refine ⟨current, ?_, rfl⟩
    rw [FP.Structural.range, if_neg (by omega), if_neg hn]
  | succ fuel ih =>
    by_cases hn : n < length p
    · let next := bodyExecution rest cfg plans profile out p initial current n h
      have hnext := bodyInvariant rest cfg plans profile out p initial current n h
      obtain ⟨final, hr, hm⟩ := ih (n+p.blockRows) next.val (by omega) hnext
      refine ⟨final, ?_, ?_⟩
      · rw [FP.Structural.range, if_neg (by omega), if_pos hn, next.property]
        exact hr
      · rw [reference, if_pos hn]
        rw [hm]
        exact congrArg (reference cfg plans profile out p initial fuel (n+p.blockRows))
          (bodyMemory rest cfg plans profile out p initial current n h)
    · refine ⟨current, ?_, ?_⟩
      · rw [FP.Structural.range, if_neg (by omega), if_neg hn]
      · simp [reference, hn]

#axiomsClean loopExecution
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
