import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFlatBody

/-! Live iterations use no-underflow derived from their actual time-indexed
pointers. There is deliberately no such claim at time zero. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatTail
open VeriTile.Triton Primitives BackwardSource BackwardSecondState BackwardFlatBody
open ForwardState (keyPointers valuePointers outputPointers)
open BackwardFirstState (keyGradientPointers)
noncomputable section
set_option maxHeartbeats 1300000
set_option linter.unusedSimpArgs false

private theorem assign_cons {d sh name} {e : Op d sh} {s : BlockState} {bounds rest}
    (hsafe : e.SafeAtR .triv bounds s)
    (htail : ∀ v, Stmt.TraceSafeListR .triv bounds rest (s.setReg name d sh v)) :
    Stmt.TraceSafeListR .triv bounds (.assign d sh name e :: rest) s := by
  refine Stmt.TraceSafeListR.cons_intro (by simpa only [Stmt.TraceSafeR] using hsafe) ?_
  intro u hu
  obtain ⟨v,_,rfl⟩ := stepStmtR_assign_inv hu
  exact htail v

private theorem retreat_safe (n stride : Nat) (r : RegionName) (name : RegName)
    (addr : Fin n → Nat) (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .ptr [n] name = some (pointers n r addr))
    (hb : ∀ j, stride ≤ addr j) :
    (Op.ptrSub .scalarR (.ref .ptr [n] name) (.constNat stride)).SafeAtR .triv bounds u := by
  rw [Op.SafeAtR.eq_def]
  refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
  intro ptrs offs hptrs hoffs i
  have ep : ptrs = pointers n r addr :=
    (Option.some.inj (by simpa only [evalOpR,hp] using hptrs)).symm
  have eo : offs = Tile.scalar stride :=
    (Option.some.inj (by simpa only [evalOpR] using hoffs)).symm
  subst ptrs
  subst offs
  change stride ≤ addr i.1
  exact hb i.1

private theorem stride_le_address (base stride t : Nat) (ht : 0 < t) :
    stride ≤ base + t*stride := by
  have hm := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt ht)
  simp only [Nat.one_mul] at hm
  omega

theorem live_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (ht : 0 < t) (u : BlockState) (hu : Ready p l initial r (t+1) u)
    (bounds : RegionBounds) :
    Stmt.TraceSafeListR .triv bounds (tail p) (afterCore p l initial r t u) := by
  dsimp only [tail,secondBody,List.drop,retreat]
  apply assign_cons
  · exact retreat_safe p.BV p.DV r.do_ "p_do" (fun j => valueAddress p l t j.val) bounds _
      (by simpa [afterCore,scatterK,scatterV,beforeStores,valuePointers] using hu.dout (by omega))
      (fun _ => stride_le_address _ _ t ht)
  intro dout
  apply assign_cons
  · exact retreat_safe p.BK p.DK r.q "p_q" (fun k => keyAddress p l t k.val) bounds _
      (by simpa [afterCore,scatterK,scatterV,beforeStores,keyPointers] using hu.q (by omega))
      (fun _ => stride_le_address _ _ t ht)
  intro q
  apply assign_cons
  · exact retreat_safe p.BK p.DK r.k "p_k" (fun k => keyAddress p l t k.val) bounds _
      (by simpa [afterCore,scatterK,scatterV,beforeStores,keyPointers] using hu.k (by omega))
      (fun _ => stride_le_address _ _ t ht)
  intro k
  apply assign_cons
  · exact retreat_safe p.BV p.DV r.v "p_v" (fun j => valueAddress p l t j.val) bounds _
      (by simpa [afterCore,scatterK,scatterV,beforeStores,valuePointers] using hu.v (by omega))
      (fun _ => stride_le_address _ _ t ht)
  intro v
  apply assign_cons
  · exact retreat_safe p.BK p.DK r.dk "p_dk" (fun k => keyGradientAddress p l t k.val) bounds _
      (by simpa [afterCore,scatterK,scatterV,beforeStores,keyGradientPointers] using hu.dk (by omega))
      (fun _ => stride_le_address _ _ t ht)
  intro dk
  apply assign_cons
  · exact retreat_safe p.BV p.DV r.dv "p_dv" (fun j => outputAddress p l t j.val) bounds _
      (by simpa [afterCore,scatterK,scatterV,beforeStores,outputPointers] using hu.dv (by omega))
      (fun _ => stride_le_address _ _ t ht)
  intro dv
  exact Stmt.TraceSafeListR.nil_intro

#axiomsClean live_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatTail
