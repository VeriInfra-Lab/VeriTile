import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSourceSafety

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety Reused FPEngine FPContext
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem query_loop (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) (bounds : RegionName → Nat)
    (hb : Bounds B D p initial bounds) (hB : 0<B) (aligned : row%B=0) :
    rangeIntSafe (engine (withDot M dot)) bounds "start_m" row p.queryContext B
      (parts B D p).body current := by
  have nonzero := Nat.ne_of_gt hB
  rw [rangeIntSafe_unfold]
  by_cases live : row<p.queryContext
  · simp only [if_neg nonzero, if_pos (by exact_mod_cast live : (row : Int) < (p.queryContext : Int))]
    rw [(FPBody.executes M dot B D p initial current row acc ctx).property]
    simpa only [Nat.cast_add] using (show _ from ⟨body M dot B D p initial current row acc ctx bounds hb aligned live,
      query_loop M dot B D p initial (FPBody.executes M dot B D p initial current row acc ctx).val
        (row+B) (FPValues.update M dot B D p initial row acc)
        (FPValues.body_context M dot B D p initial current row acc ctx) bounds hb hB
        (by simpa using aligned)⟩)
  · simp only [if_neg nonzero, if_neg (by exact_mod_cast live : ¬ (row : Int) < (p.queryContext : Int))]
termination_by p.queryContext-row
decreasing_by omega

theorem source_safe (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds B D p s bounds) (hB : 0<B) (keyOffsetBound : s.pids 0*B < 2147483648) :
    TraceSafe (engine (withDot M dot)) bounds (FPEngine.kernel B D p) s := by
  let start := FPEntry.executes M dot B D p s
  obtain ⟨middle,ran,row,ctx⟩ := FPLoop.executes M dot B D p s start.val (entry_context M dot B D p s keyOffsetBound)
  have low : (parts B D p).start=.ref .int [] "lo" := rfl
  have high : (parts B D p).stop=.ref .int [] "hi" := rfl
  have stride : (parts B D p).stride=.constNat B := rfl
  have hlo : start.val.regs .int [] "lo"=some (fun _ => 0) := (FPLoop.entry_limits M dot B D p s).1
  have hhi : start.val.regs .int [] "hi"=some (fun _ => (p.queryContext : Int)) := (FPLoop.entry_limits M dot B D p s).2
  have loopRun : step (engine (withDot M dot))
      (.forRangeInt "start_m" (parts B D p).start (parts B D p).stop
        (parts B D p).stride (parts B D p).body) start.val=some middle := by
    simpa only [step,low,high,stride,evalOp_unfold,hlo,hhi,Option.bind,Option.bind_some] using ran
  unfold TraceSafe
  rw [FPOutput.source_parts,List.append_assoc,traceSafeList_append]
  refine ⟨prelude M dot B D p s bounds hb keyOffsetBound,?_⟩
  intro next hnext
  have eqNext : next=start.val := Option.some.inj (hnext.symm.trans start.property)
  subst next
  apply traceSafeList_cons_of_step (t:=middle) (executed:=loopRun)
  · simp only [stepSafe_unfold,low,high,stride,safeOp_unfold,evalOp_unfold,hlo,hhi,true_and]
    exact query_loop M dot B D p s start.val 0 (fun _ => zero M)
      (entry_context M dot B D p s keyOffsetBound) bounds hb hB (by simp)
  · exact output M dot B D p s middle row (FPLoop.accumulated M dot B D p s) ctx bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
