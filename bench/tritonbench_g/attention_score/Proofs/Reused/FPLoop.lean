import bench.tritonbench_g.attention_score.Proofs.Reused.FPValues
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPLoop
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPEngine FPContext FPValues
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem signed_range_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg idx .int [] (fun _ => (n:Int))) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t)
    (h : I cur initial s) :
    ∃ final, FP.Structural.rangeInt M idx (cur:Int) (stop:Int) stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  by_cases hz : stride = 0
  · exact ⟨s, by simp [FP.Structural.rangeInt, hz], by simpa [foldRange, hz] using hdone cur initial s h⟩
  · by_cases hc : cur < stop
    · obtain ⟨next, he, hn⟩ := hstep cur initial s h hc
      obtain ⟨final, hf, hd⟩ := signed_range_values M idx body update I Done
        (cur+stride) stop stride (update cur initial) next hstep hdone hn
      refine ⟨final, ?_, ?_⟩
      · rw [FP.Structural.rangeInt]
        simpa only [hz, show (cur:Int)<(stop:Int) by omega, if_false, if_true, he, Option.bind_some,Int.natCast_add] using hf
      · rw [foldRange]
        simpa only [hz, hc, if_false, if_true] using hd
    · exact ⟨s, by simp [FP.Structural.rangeInt, hz, show ¬(cur:Int)<(stop:Int) by omega], by simpa [foldRange, hz, hc] using hdone cur initial s h⟩
termination_by stop-cur
decreasing_by omega


/-- Ordered query-block recurrence, with no execution or output-memory reference. -/
def accumulated (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) : Fin B → α :=
  foldRange (update M dot B D p s) 0 p.queryContext B (fun _ => zero M)

theorem executes (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (ctx : Context M B D p initial 0 (fun _ => zero M) current) :
    ∃ t, rangeInt (engine (withDot M dot)) "start_m" 0 (p.queryContext:Int) B (parts B D p).body current=some t ∧
      ∃ row, Context M B D p initial row (accumulated M dot B D p initial) t := by
  exact signed_range_values (engine (withDot M dot)) "start_m" (parts B D p).body (update M dot B D p initial)
    (fun row acc s => Context M B D p initial row acc s)
    (fun acc s => ∃ row, Context M B D p initial row acc s) 0 p.queryContext B _ current
    (by
      intro row acc s h _
      let t := FPBody.executes M dot B D p initial s row acc h
      exact ⟨t.val,t.property,body_context M dot B D p initial s row acc h⟩)
    (by intro row acc s h; exact ⟨row,h⟩) ctx

theorem entry_limits (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α) :
    (FPEntry.executes M dot B D p s).val.regs .nat [] "lo"=some (fun _ => 0) ∧
    (FPEntry.executes M dot B D p s).val.regs .int [] "hi"=some (fun _ => (p.queryContext:Int)) := by
  by_cases hn : p.evenN = Bool.true
  all_goals simp [FPEntry.executes,hn]

#specNonCircular accumulated avoiding [run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPLoop
