import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.SavedInputs
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
noncomputable section
open Real
open scoped BigOperators
set_option maxHeartbeats 10000000
theorem ops_zero : ops.zero=(0:ℝ) := rfl
theorem ops_add (x y : ℝ) : ops.add x y=x+y := rfl
theorem ops_sub (x y : ℝ) : ops.sub x y=x-y := rfl
theorem ops_mul (x y : ℝ) : ops.mul x y=x*y := rfl
def grad (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (b v t : Nat) :=
  Math.gradient scale (column s f m b v) (incoming m b v) (terminal s f m b v) s.T t

theorem grad_x (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (b v t : Nat) :
    (grad s f scale m b v t).x=(column s f m b v).beta t*
      Math.dot (enhanced s f scale m b v t) ((column s f m b v).k t) := by
  have he : s.T-(t+1)=s.T-1-t := by omega
  simp only [grad,Math.gradient,Math.carry,Math.pullback,he]
  rfl
theorem grad_beta (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (b v t : Nat) :
    (grad s f scale m b v t).beta=
      Math.residual (Math.state (column s f m b v) t) ((column s f m b v).k t) ((column s f m b v).x t)*
      Math.dot (enhanced s f scale m b v t) ((column s f m b v).k t) := by
  have he : s.T-(t+1)=s.T-1-t := by omega
  simp only [grad,Math.gradient,Math.carry,Math.pullback,he]
  rfl
theorem grad_q (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (b v t : Nat) (i : Fin s.K) :
    (grad s f scale m b v t).q i=scale*incoming m b v t*Math.state (column s f m b v) (t+1) i := rfl
theorem grad_k (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (b v t : Nat) (i : Fin s.K) :
    (grad s f scale m b v t).k i=
      (column s f m b v).beta t*
        Math.residual (Math.state (column s f m b v) t) ((column s f m b v).k t) ((column s f m b v).x t)*
        enhanced s f scale m b v t i-(grad s f scale m b v t).x*Math.state (column s f m b v) t i := by
  have he : s.T-(t+1)=s.T-1-t := by omega
  simp only [grad,Math.gradient,Math.carry,Math.pullback,he,Real.enhanced,Real.adjoint]

theorem raw_dv (s : Shape) (f : Flags) (gf gb : Trees) (hg : (gb.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (b iv j t : Nat) (hv : iv*s.BBV+j<s.V) :
    Reverse.rawDv s f ops gb scale (saved s f gf scale m) b iv t j=
      Math.dot (enhanced s f scale m b (iv*s.BBV+j) t) ((column s f m b (iv*s.BBV+j)).k t) := by
  rw [raw_dv_math s f gb hg scale _ b iv j t hv,saved_enhanced]
  congr 1
  funext i
  exact saved_stable s f gf scale m .k (by trivial) _ _ _
theorem dv (s : Shape) (f : Flags) (gf gb : Trees) (hg : (gb.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (b iv j t : Nat) (hv : iv*s.BBV+j<s.V) :
    Reverse.dv s f ops gb scale (saved s f gf scale m) b iv t j=(grad s f scale m b (iv*s.BBV+j) t).x := by
  change Reverse.rawDv s f ops gb scale (saved s f gf scale m) b iv t j*betaLoad s f ops _ b t iv s.BBV j=_
  rw [raw_dv s f gf gb hg scale m b iv j t hv,grad_x]
  simp only [betaLoad,valLoad,load,hv,if_true,saved_stable s f gf scale m .beta (by trivial),column]
  ring
theorem dbeta (s : Shape) (f : Flags) (gf gb : Trees)
    (hgf : (gf.key s.BK).Legal s.BK) (hgb : (gb.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (b iv j t : Nat) (hb : b<s.BH) (ht : t<s.T) (hv : iv*s.BBV+j<s.V) :
    Reverse.dbeta s f ops gb scale (saved s f gf scale m) b iv t j=(grad s f scale m b (iv*s.BBV+j) t).beta := by
  change Reverse.rawDv s f ops gb scale (saved s f gf scale m) b iv t j*valLoad s ops _ .v b t iv s.BBV j=_
  rw [raw_dv s f gf gb hgb scale m b iv j t hv,grad_beta]
  simp only [valLoad,load,hv,if_true,saved_residual s f gf hgf scale m b t (iv*s.BBV+j) hb ht hv]
  ring

theorem tree_zero (tr : Tree) : tr.eval ops (fun _ => (0:ℝ))=0 := by
  induction tr <;> simp_all [Tree.eval,ops_zero,ops_add]
theorem reverse_inactive (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (b iv j : Nat) (hv : ¬iv*s.BBV+j<s.V) (n i : Nat) :
    Reverse.h s f ops g scale m b iv n i j=0 := by
  induction n generalizing i with
  | zero => simp [Reverse.h,seed,load,hv,ops_zero,ops_add,ops_mul,ops_sub]
  | succ n ih =>
    simp only [Reverse.h,valLoad,load,hv,if_false,ops_zero,ops_add,ops_mul,ops_sub,mul_zero,add_zero,ih,zero_mul,tree_zero,zero_sub,neg_zero]

theorem enhanced_inactive (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (b iv j : Nat) (hv : ¬iv*s.BBV+j<s.V) (t i : Nat) :
    Reverse.enhanced s f ops g scale m b iv t i j=0 := by
  simp [Reverse.enhanced,reverse_inactive s f g scale m b iv j hv,valLoad,load,hv,ops_zero,ops_add,ops_mul,ops_sub]
theorem dv_inactive (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (b iv j : Nat) (hv : ¬iv*s.BBV+j<s.V) (t : Nat) :
    Reverse.dv s f ops g scale m b iv t j=0 := by
  simp [Reverse.dv,Reverse.rawDv,enhanced_inactive s f g scale m b iv j hv,ops_zero,ops_add,ops_mul,ops_sub,tree_zero]
theorem dbeta_inactive (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (b iv j : Nat) (hv : ¬iv*s.BBV+j<s.V) (t : Nat) :
    Reverse.dbeta s f ops g scale m b iv t j=0 := by
  simp [Reverse.dbeta,valLoad,load,hv,ops_zero,ops_add,ops_mul,ops_sub]
theorem recompute_inactive (s : Shape) (f : Flags) (m : Memory ℝ)
    (b iv j : Nat) (hv : ¬iv*s.BBV+j<s.V) (n i : Nat) :
    Recompute.h s f ops m b iv n i j=0 := by
  induction n with
  | zero => simp [Recompute.h,seed,load,hv,ops_zero,ops_add,ops_mul,ops_sub]
  | succ n ih => simp [Recompute.h,valLoad,load,hv,ops_zero,ops_add,ops_mul,ops_sub,ih]

#axiomsClean dv
#axiomsClean dbeta
#axiomsClean reverse_inactive
#axiomsClean dv_inactive
#axiomsClean recompute_inactive
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
