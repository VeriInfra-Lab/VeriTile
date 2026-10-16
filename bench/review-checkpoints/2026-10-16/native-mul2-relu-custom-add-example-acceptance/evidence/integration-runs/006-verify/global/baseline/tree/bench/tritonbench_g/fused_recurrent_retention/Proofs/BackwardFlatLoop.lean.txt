import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFlatTail

/-! Complete reverse-loop finite-memory execution. Strict correspondence is
maintained while another iteration remains; the actual final pointer tail
executes with observational agreement of memory, ids, grid and undef. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatLoop
open VeriTile.Triton BackwardSource BackwardFlatBody
noncomputable section
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

theorem executes (A : FlatAlloc) (ha : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (hd : BackwardReal.Disjoint r)
    (hb : BackwardSafety.Bounds p l r hasSeed A.extent)
    (u final : BlockState) (hu : BackwardSecondLoop.Holds p l initial r hasSeed p.T u)
    (hz : u.undef = fun _ _ => 0)
    (he : stepForRangeAux "_t" 0 p.T 1 (secondBody p) u = some final) :
    ∃ flat, stepForRangeAux "_t" 0 p.T 1 (A.flattenStmts (secondBody p))
      (A.flattenState u) = some flat ∧
      flat.ObsAgree (A.flattenState final) ∧ RegsDefEq flat final := by
  let P := fun c => BackwardSecondLoop.Holds p l initial r hasSeed (p.T-c)
  have hstep : ∀ c s, c < p.T → P c s →
      Stmt.TraceSafeListR .triv A.extent (core p)
        (s.setReg "_t" .nat [] (Tile.scalar c)) ∧
      (∃ v, stepStmtsR .triv (core p ++ tail p)
          (s.setReg "_t" .nat [] (Tile.scalar c)) = some v ∧ P (c+1) v) ∧
      (c+1 < p.T → ∀ mid,
        stepStmtsR .triv (core p) (s.setReg "_t" .nat [] (Tile.scalar c)) = some mid →
        Stmt.TraceSafeListR .triv A.extent (tail p) mid) := by
    intro c s hct hs
    have hrem : p.T-c = (p.T-(c+1))+1 := by omega
    have hs' : BackwardSecondLoop.Holds p l initial r hasSeed ((p.T-(c+1))+1) s := by
      simpa only [P,hrem] using hs
    have hi := BackwardSafety.holds_set_loop p l initial r hasSeed _ c s hs'
    refine ⟨core_safe p l initial r hasSeed A.extent hb _ (by omega) _ hi,?_,?_⟩
    · obtain ⟨v,hv,hnext⟩ := BackwardSecondLoop.iteration p l initial r hasSeed
        (p.T-(c+1)) c (by omega) s hd.toDisjoint hs'
      refine ⟨v,?_,hnext⟩
      simpa only [←split,stepStmtsR_triv] using hv
    · intro hn mid hm
      rw [stepStmtsR_triv,core_executes p l initial r hasSeed _ (by omega) _ hi] at hm
      cases hm
      exact BackwardFlatTail.live_safe p l initial r _ (by omega) _ hi.ready A.extent
  have hrun := A.stepForRangeAuxR_flatten_deadPtrTail ha hc .triv "_t" p.T 1
    (core p) (tail p) (tail_shape p) (core_flatten_ok p) P hstep 0 u
    (by simpa only [P,Nat.sub_zero] using hu) hz final
    (by simpa only [←split,stepForRangeAuxR_triv] using he)
  simpa only [←split,stepForRangeAuxR_triv] using hrun

#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatLoop
