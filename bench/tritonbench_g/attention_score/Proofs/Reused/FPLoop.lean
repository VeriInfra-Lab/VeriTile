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

/-- Ordered query-block recurrence, with no execution or output-memory reference. -/
def accumulated (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) : Fin B → α :=
  foldRange (update M dot B D p s) 0 p.queryContext B (fun _ => zero M)

theorem executes (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (ctx : Context M B D p initial 0 (fun _ => zero M) current) :
    ∃ t, range (engine (withDot M dot)) "start_m" 0 p.queryContext B (parts B D p).body current=some t ∧
      ∃ row, Context M B D p initial row (accumulated M dot B D p initial) t := by
  exact range_values (engine (withDot M dot)) "start_m" (parts B D p).body (update M dot B D p initial)
    (fun row acc s => Context M B D p initial row acc s)
    (fun acc s => ∃ row, Context M B D p initial row acc s) 0 p.queryContext B _ current
    (by
      intro row acc s h _
      let t := FPBody.executes M dot B D p initial s row acc h
      exact ⟨t.val,t.property,body_context M dot B D p initial s row acc h⟩)
    (by intro row acc s h; exact ⟨row,h⟩) ctx

theorem entry_limits (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α) :
    (FPEntry.executes M dot B D p s).val.regs .nat [] "lo"=some (fun _ => 0) ∧
    (FPEntry.executes M dot B D p s).val.regs .nat [] "hi"=some (fun _ => p.queryContext) := by
  by_cases hn : p.evenN = Bool.true
  all_goals simp [FPEntry.executes,hn]

#specNonCircular accumulated avoiding [run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPLoop
