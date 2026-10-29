import VeriTile.Triton.Memory.SignedRangeTrace
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyParts

/-! Actual query-loop trace safety. The execution invariant supplies Q/M
pointers, and each visited block is bounded by the proved ceiling trip count.
No condition is imposed on numerical outputs or on unexecuted iterations. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyLoop
open VeriTile.Triton
open VeriTile.Triton.Memory.AccessHelpers
set_option maxHeartbeats 1000000
noncomputable section

theorem correct (bounds : RegionBounds) (c : RealLoop.Config) (initial s : BlockState)
    (stop : Nat) (hB : 0 < c.B) (hi : RealLoop.Invariant c initial 0 s)
    (hbQ : ∀ j, j < RealLoop.tripCount stop c.B →
      BlockBounds bounds (RealLoads.queryPointer c.Q c.B c.D c.N c.baseQ c.sqm c.sqk (j*c.B))
        (RealLoads.checks c.evenM))
    (hbM : ∀ row, row < c.N → c.baseM+row < bounds c.M) :
    Stmt.forRangeIntTraceSafe bounds "start_m" 0 stop c.B c.body s := by
  let count := RealLoop.tripCount stop c.B
  let P := fun (row : Int) st => ∃ j : Nat, row = ((j*c.B : Nat) : Int) ∧ j ≤ count ∧ RealLoop.Invariant c initial j st
  have hu : stop ≤ count*c.B := (RealLoop.tripCount_bounds stop c.B hB).1
  have visited : ∀ j, j ≤ count → j*c.B < stop → j < count := by
    intro j hj hrow
    by_contra hn
    have he : j = count := by omega
    rw [he] at hrow
    omega
  apply TraceProofs.range_int_safe (P := P)
  · intro row st hrow hp
    obtain ⟨j,rfl,hj,hregs⟩ := hp
    have hjlt := visited j hj (by exact_mod_cast hrow)
    apply RealSafetyParts.body_safe bounds _ c.Q c.M c.B c.D c.N c.baseQ c.baseM c.sqm c.sqk
      (j*c.B) c.offset c.size c.window c.complement c.evenM c.evenN c.candidate
      (by simp [hregs.query]) (by simp [hregs.normalizer]) (hbQ j hjlt)
    · simp
    · intro i hrow
      simpa only [Nat.add_assoc] using hbM (j*c.B+i.val) hrow
  · intro row st t hrow hp he
    obtain ⟨j,rfl,hj,hregs⟩ := hp
    have hjlt := visited j hj (by exact_mod_cast hrow)
    obtain ⟨u,hu,iu,_⟩ := RealLoop.iteration c initial st j hB hregs
    have hut : u = t := Option.some.inj (hu.symm.trans he)
    subst t
    exact ⟨j+1, by simp [Nat.add_mul, Nat.cast_add], by omega, iu⟩
  · exact ⟨0, by simp, Nat.zero_le _, hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyLoop
