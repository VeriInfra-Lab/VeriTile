import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.BackwardLocal
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
noncomputable section
open Real
open scoped BigOperators
set_option maxHeartbeats 12000000
def tileSum (s : Shape) (iv : Nat) (a : Nat → ℝ) :=
  ∑ j∈Finset.range s.BBV,if iv*s.BBV+j<s.V then a (iv*s.BBV+j) else 0

theorem query_tile (s : Shape) (f : Flags) (gf gb : Trees)
    (hgf : (gf.key s.BK).Legal s.BK) (hq : (gb.query s.BBV).Legal s.BBV)
    (scale : ℝ) (m : Memory ℝ) (b iv t : Nat) (i : Fin s.K) (hb : b<s.BH) (ht : t<s.T) :
    Recompute.output s f ops gb scale (BackwardResults.mid s f ops gb scale (saved s f gf scale m) b iv) b iv t i.val=
      tileSum s iv (fun v => (grad s f scale m b v t).q i) := by
  change (gb.query s.BBV).eval ops (fun j =>
    Recompute.h s f ops _ b iv (t+1) i.val j*valLoad s ops _ .do_ b t iv s.BBV j)*scale=_
  rw [tree_sum _ _ hq,Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hv : iv*s.BBV+j<s.V
  · rw [if_pos hv,recompute_h s f gf gb hgf scale m b iv j hb hv (t+1) (by omega) i]
    simp only [valLoad,load,hv,if_true,mid_readonly s f gb scale _ b iv .do_ (by trivial),
      saved_stable s f gf scale m .do_ (by trivial),grad_q,incoming]
    ring
  · simp [valLoad,load,hv,ops]

theorem key_tile (s : Shape) (f : Flags) (gf gb : Trees)
    (hgf : (gf.key s.BK).Legal s.BK) (hgb : (gb.key s.BK).Legal s.BK)
    (hdk : (gb.value s.BBV).Legal s.BBV) (hc : (gb.correction s.BBV).Legal s.BBV)
    (scale : ℝ) (m : Memory ℝ) (b iv t : Nat) (i : Fin s.K) (hb : b<s.BH) (ht : t<s.T) :
    Recompute.corrected s f ops gb (BackwardResults.mid s f ops gb scale (saved s f gf scale m) b iv) b iv t i.val=
      tileSum s iv (fun v => (grad s f scale m b v t).k i) := by
  change keyLoad s ops _ .dk (iv*s.BH+b) t i.val-
    (gb.correction s.BBV).eval ops (fun j => valLoad s ops _ .dv b t iv s.BBV j*Recompute.h s f ops _ b iv t i.val j)=_
  simp only [keyLoad,load,i.isLt,if_true]
  rw [mid_dk s f gb scale _ b iv t i ht]
  change (gb.value s.BBV).eval ops (fun j => Reverse.enhanced s f ops gb scale _ b iv t i.val j*
      (valLoad s ops _ .v b t iv s.BBV j*betaLoad s f ops _ b t iv s.BBV j))-
    (gb.correction s.BBV).eval ops (fun j => valLoad s ops _ .dv b t iv s.BBV j*Recompute.h s f ops _ b iv t i.val j)=_
  rw [tree_sum _ _ hdk,tree_sum _ _ hc,←Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  have hj' : j<s.BBV := Finset.mem_range.mp hj
  by_cases hv : iv*s.BBV+j<s.V
  · rw [if_pos hv,enhanced_math s f gb hgb scale _ b iv j t hv i,saved_enhanced,
      recompute_h s f gf gb hgf scale m b iv j hb hv t (by omega) i]
    simp only [valLoad,betaLoad,load,hv,if_true]
    rw [mid_dv s f gb scale _ b iv t j ht hj' hv,dv s f gf gb hgb scale m b iv j t hv,
      saved_residual s f gf hgf scale m b t (iv*s.BBV+j) hb ht hv]
    simp only [saved_stable s f gf scale m .beta (by trivial),grad_k,column]
    ring
  · simp [valLoad,load,hv,ops]

theorem beta_tile (s : Shape) (f : Flags) (gf gb : Trees)
    (hgf : (gf.key s.BK).Legal s.BK) (hgb : (gb.key s.BK).Legal s.BK) (hbta : (gb.beta s.BBV).Legal s.BBV)
    (scale : ℝ) (m : Memory ℝ) (b iv t : Nat) (hb : b<s.BH) (ht : t<s.T) :
    (gb.beta s.BBV).eval ops (fun j => Reverse.dbeta s f ops gb scale (saved s f gf scale m) b iv t j)=
      tileSum s iv (fun v => (grad s f scale m b v t).beta) := by
  rw [tree_sum _ _ hbta]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hv : iv*s.BBV+j<s.V
  · rw [if_pos hv,dbeta s f gf gb hgf hgb scale m b iv j t hb ht hv]
  · rw [if_neg hv,dbeta_inactive s f gb scale _ b iv j hv t]

#axiomsClean query_tile
#axiomsClean key_tile
#axiomsClean beta_tile
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
