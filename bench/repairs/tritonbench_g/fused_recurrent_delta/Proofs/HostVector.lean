import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.HostResults
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
noncomputable section
open Real
open scoped BigOperators
set_option maxHeartbeats 10000000

theorem grid_beta_vector (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b iv t v : Nat) (hb : b<s.BH) (hi : iv<s.BNV)
    (ht : t<s.T) (hv : v<s.V) (hf : f.vectorBeta=Bool.true) :
    backwardGrid s f ops gb scale (saved s f gf scale m) ⟨.dbeta,iv*s.BH+b,t,v⟩=
      if iv=v/s.BBV then (grad s f scale m b v t).beta else 0 := by
  by_cases hi' : iv=v/s.BBV
  · subst iv
    obtain ⟨ho,hj,he⟩:=Geometry.owner_exists s.V s.BBV v (Geometry.tile_pos s).2.2 hv
    have hlo : (v/s.BBV)*s.BBV≤v := by omega
    have hhi : v<min ((v/s.BBV+1)*s.BBV) s.V := by
      apply lt_min _ hv
      rw [Nat.add_mul]
      omega
    have hsub : v-(v/s.BBV)*s.BBV=v%s.BBV := by omega
    rw [BackwardResults.dbeta_value s f ops gb hl.correction scale _ b (v/s.BBV) t v hb hi ht hf]
    simp only [hlo,hhi,and_self,if_true,hsub]
    change Reverse.dbeta s f ops gb scale _ b (v/s.BBV) t (v%s.BBV)=_
    rw [dbeta s f gf gb hl.fkey hl.bkey scale m b (v/s.BBV) (v%s.BBV) t hb ht (by simpa [he] using hv),he]
  · rw [if_neg hi']
    apply BackwardResults.dbeta_off_tile_zero s f ops gb hl.correction scale _ b iv t v hb hi ht hv hf
    intro hc
    exact hi' (Geometry.owner_unique s.BBV iv v hc.1 (lt_of_lt_of_le hc.2 (Nat.min_le_left _ _)))

theorem return_beta_vector (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b t v : Nat) (hb : b<s.BH) (ht : t<s.T) (hv : v<s.V) (hf : f.vectorBeta=Bool.true) :
    finish s f gf gb scale m ⟨.retbeta,b,t,v⟩=(grad s f scale m b v t).beta := by
  rw [finish,host_beta s f ops gb _ b t v hb ht (by simpa [hf] using hv)]
  change (gb.hostBeta s.BNV).eval ops _=_
  rw [tree_sum _ _ hl.hostBeta]
  have he : (∑ iv∈Finset.range s.BNV,backwardGrid s f ops gb scale (saved s f gf scale m) ⟨.dbeta,iv*s.BH+b,t,v⟩)=
      ∑ iv∈Finset.range s.BNV,if iv=v/s.BBV then (grad s f scale m b v t).beta else 0 := by
    apply Finset.sum_congr rfl
    intro iv hi
    exact grid_beta_vector s f gf gb hl scale m b iv t v hb (Finset.mem_range.mp hi) ht hv hf
  rw [he]
  have ho := (Geometry.owner_exists s.V s.BBV v (Geometry.tile_pos s).2.2 hv).1
  change v/s.BBV<s.BNV at ho
  simp [Finset.sum_ite_eq',Finset.mem_range,ho]

theorem return_h0 (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b v : Nat) (i : Fin s.K) (hb : b<s.BH) (hv : v<s.V) (hf : f.initialGrad=Bool.true) :
    finish s f gf gb scale m ⟨.dh0,b,i.val,v⟩=
      Math.carry scale (column s f m b v) (incoming m b v) (terminal s f m b v) s.T 0 i := by
  obtain ⟨ho,hj,he⟩:=Geometry.owner_exists s.V s.BBV v (Geometry.tile_pos s).2.2 hv
  have hh : finish s f gf gb scale m ⟨.dh0,b,i.val,v⟩=
      backwardGrid s f ops gb scale (saved s f gf scale m) ⟨.dh0,b,i.val,v⟩ := by
    simp [finish,hostReduce,write]
  rw [hh]
  have hr:=BackwardResults.dh0_value s f ops gb hl.correction scale (saved s f gf scale m) b (v/s.BBV) i.val (v%s.BBV) hb ho i.isLt hj (by simpa [he] using hv) hf
  rw [he] at hr
  rw [hr]
  change Reverse.h s f ops gb scale _ b (v/s.BBV) s.T i.val (v%s.BBV)=_
  rw [reverse_h_math s f gb hl.bkey scale _ b (v/s.BBV) (v%s.BBV) (by simpa [he] using hv) s.T i,
    saved_adjoint,he]
  simp [adjoint,Math.carry]

#axiomsClean grid_beta_vector
#axiomsClean return_beta_vector
#axiomsClean return_h0
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
