import bench.tritonbench_g.attention_score.Proofs.Reused.RealSafetyParts

/-! Actual query-loop trace safety. The execution invariant supplies Q/M
pointers, and each visited block is bounded by the proved ceiling trip count.
No condition is imposed on numerical outputs or on unexecuted iterations. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyLoop
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
set_option maxHeartbeats 1000000
noncomputable section

theorem correct (bounds : RegionBounds) (c : RealLoop.Config) (initial s : BlockState)
    (stop : Nat) (hB : 0 < c.B) (hi : RealLoop.Invariant c initial 0 s)
    (hbQ : ∀ j, j < RealLoop.tripCount stop c.B →
      BlockBounds bounds (RealLoads.queryPointer c.Q c.B c.D c.N c.baseQ c.sqm c.sqk (j*c.B))
        (RealLoads.checks c.evenM))
    (hbM : ∀ row, row < c.N → c.baseM+row < bounds c.M) :
    Stmt.forRangeIntTraceSafe bounds "start_m" 0 (stop:Int) c.B c.body s := by
  let count := RealLoop.tripCount stop c.B
  have hu : stop ≤ count*c.B := (RealLoop.tripCount_bounds stop c.B hB).1
  have hl := (RealLoop.tripCount_bounds stop c.B hB).2
  have remaining (left done : Nat) (st : BlockState) (total : done+left=count)
      (regs : RealLoop.Invariant c initial done st) :
      Stmt.forRangeIntTraceSafe bounds "start_m" ((done*c.B:Nat):Int) (stop:Int) c.B c.body st := by
    induction left generalizing done st with
    | zero =>
        have he : done=count := by omega
        rw [Stmt.forRangeIntTraceSafe,if_neg (by omega),if_neg (by exact_mod_cast (show ¬done*c.B<stop from by rw [he];omega))]
        trivial
    | succ left ih =>
        have hd : done<count := by omega
        have hdstop : done*c.B<stop := by
          rcases hl with hz | hl
          · omega
          · exact lt_of_le_of_lt (Nat.mul_le_mul_right c.B (by omega : done≤count-1)) hl
        obtain ⟨next,he,regs',_⟩ := RealLoop.iteration c initial st done hB regs
        rw [Stmt.forRangeIntTraceSafe,if_neg (by omega),if_pos (by exact_mod_cast hdstop),he]
        constructor
        · apply RealSafetyParts.body_safe bounds _ c.Q c.M c.B c.D c.N c.baseQ c.baseM c.sqm c.sqk
            (done*c.B) c.offset c.size c.window c.complement c.evenM c.evenN c.candidate
            (by simp [regs.query]) (by simp [regs.normalizer]) (hbQ done hd)
          · simp
          · intro i hr
            simpa only [Nat.add_assoc] using hbM (done*c.B+i.val) hr
        · have index : ((done*c.B:Nat):Int)+(c.B:Int)=(((done+1)*c.B:Nat):Int) := by push_cast;ring
          rw [index]
          exact ih (done+1) next (by omega) regs'
  simpa using remaining count 0 s (by omega) hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyLoop
