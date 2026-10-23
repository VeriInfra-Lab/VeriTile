import bench.tritonbench_g.parallel_attention.Proofs.FPForwardGraph
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section

/-- A numerical recurrence may be carried through a source loop without any
associativity or reduction law. The premise proves one actual body execution;
the conclusion follows the original dynamic range indices exactly. -/
theorem range_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg idx .nat [] (fun _ => n)) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t)
    (h : I cur initial s) :
    ∃ final, FP.Structural.range M idx cur stop stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  by_cases hz : stride = 0
  · exact ⟨s, by simp [FP.Structural.range, hz], by simpa [foldRange, hz] using hdone cur initial s h⟩
  · by_cases hc : cur < stop
    · obtain ⟨next, he, hn⟩ := hstep cur initial s h hc
      obtain ⟨final, hf, hd⟩ := range_values M idx body update I Done
        (cur+stride) stop stride (update cur initial) next hstep hdone hn
      refine ⟨final, ?_, ?_⟩
      · rw [FP.Structural.range]
        simpa only [hz, hc, if_false, if_true, he, Option.bind_some] using hf
      · rw [foldRange]
        simpa only [hz, hc, if_false, if_true] using hd
    · exact ⟨s, by simp [FP.Structural.range, hz, hc], by simpa [foldRange, hz, hc] using hdone cur initial s h⟩
termination_by stop-cur
decreasing_by omega

#axiomsClean range_values
end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
