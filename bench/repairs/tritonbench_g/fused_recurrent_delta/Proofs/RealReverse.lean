import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.RealForward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Reverse
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Real
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 8000000
def incoming (m : Memory ℝ) (bh v : Nat) (t : Nat) := m ⟨.do_,bh,t,v⟩
def terminal (s : Shape) (f : Flags) (m : Memory ℝ) (bh v : Nat) : Math.Vec s.K :=
  fun i => if f.terminalGrad then m ⟨.dht,bh,i.val,v⟩ else 0
def adjoint (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (bh v n : Nat) :=
  Math.reverse scale (column s f m bh v) (incoming m bh v) (terminal s f m bh v) s.T n
def enhanced (s : Shape) (f : Flags) (scale : ℝ) (m : Memory ℝ) (bh v t : Nat) : Math.Vec s.K :=
  fun i => adjoint s f scale m bh v (s.T-1-t) i+scale*incoming m bh v t*(column s f m bh v).q t i

theorem reverse_h_math (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh iv j : Nat) (hj : iv*s.BBV+j<s.V) (n : Nat) (i : Fin s.K) :
    Reverse.h s f ops g scale m bh iv n i.val j=adjoint s f scale m bh (iv*s.BBV+j) n i := by
  induction n generalizing i with
  | zero => simp [Reverse.h,seed,load,ops,adjoint,Math.reverse,terminal,i.isLt,hj]
  | succ n ih =>
    let t:=s.T-1-n
    let E : Math.Vec s.K:=fun k => adjoint s f scale m bh (iv*s.BBV+j) n k+
      scale*incoming m bh (iv*s.BBV+j) t*(column s f m bh (iv*s.BBV+j)).q t k
    have eh (k : Fin s.K) : Reverse.h s f ops g scale m bh iv n k.val j+
        (keyLoad s ops m .q bh t k.val*scale)*valLoad s ops m .do_ bh t iv s.BBV j=E k := by
      rw [ih k]
      simp only [keyLoad,valLoad,load,k.isLt,hj,if_true,E,incoming,column]
      ring
    have sumEq : (g.key s.BK).eval ops (fun k =>
        (Reverse.h s f ops g scale m bh iv n k j+
          (keyLoad s ops m .q bh t k*scale)*valLoad s ops m .do_ bh t iv s.BBV j)*keyLoad s ops m .k bh t k)=
      Math.dot E ((column s f m bh (iv*s.BBV+j)).k t) := by
      rw [key_sum s g hg]
      apply Finset.sum_congr rfl
      intro k _
      rw [eh k]
      rfl
    change (Reverse.h s f ops g scale m bh iv n i.val j+
      (keyLoad s ops m .q bh t i.val*scale)*valLoad s ops m .do_ bh t iv s.BBV j)-
      keyLoad s ops m .k bh t i.val*
        ((g.key s.BK).eval ops (fun k =>
          (Reverse.h s f ops g scale m bh iv n k j+
           (keyLoad s ops m .q bh t k*scale)*valLoad s ops m .do_ bh t iv s.BBV j)*keyLoad s ops m .k bh t k)*
          betaLoad s f ops m bh t iv s.BBV j)=_
    rw [eh i,sumEq]
    change E i-keyLoad s ops m .k bh t i.val*(Math.dot E ((column s f m bh (iv*s.BBV+j)).k t)*betaLoad s f ops m bh t iv s.BBV j)=_
    simp only [adjoint,Math.reverse,Math.pullback]
    change E i-keyLoad s ops m .k bh t i.val*(Math.dot E ((column s f m bh (iv*s.BBV+j)).k t)*betaLoad s f ops m bh t iv s.BBV j)=
      E i-((column s f m bh (iv*s.BBV+j)).beta t*Math.dot E ((column s f m bh (iv*s.BBV+j)).k t))*
        (column s f m bh (iv*s.BBV+j)).k t i
    simp [keyLoad,valLoad,betaLoad,load,column,i.isLt,hj]
    ring

theorem enhanced_math (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh iv j t : Nat) (hj : iv*s.BBV+j<s.V) (i : Fin s.K) :
    Reverse.enhanced s f ops g scale m bh iv t i.val j=enhanced s f scale m bh (iv*s.BBV+j) t i := by
  change Reverse.h s f ops g scale m bh iv (s.T-1-t) i.val j+
    (keyLoad s ops m .q bh t i.val*scale)*valLoad s ops m .do_ bh t iv s.BBV j=_
  rw [reverse_h_math s f g hg scale m bh iv j hj (s.T-1-t) i]
  simp [enhanced,keyLoad,valLoad,load,i.isLt,hj,incoming,column]
  ring
theorem raw_dv_math (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh iv j t : Nat) (hj : iv*s.BBV+j<s.V) :
    Reverse.rawDv s f ops g scale m bh iv t j=
      Math.dot (enhanced s f scale m bh (iv*s.BBV+j) t) ((column s f m bh (iv*s.BBV+j)).k t) := by
  change (g.key s.BK).eval ops (fun i => Reverse.enhanced s f ops g scale m bh iv t i j*keyLoad s ops m .k bh t i)=_
  rw [key_sum s g hg]
  apply Finset.sum_congr rfl
  intro i _
  rw [enhanced_math s f g hg scale m bh iv j t hj i]
  rfl

/-- The adjoint recurrence is independent of pristine v and H0. This lets
the eventual full proof use the original primal while the source reloads
its overwritten residual tensor. -/
theorem adjoint_primal_independent {K : Nat} (s : ℝ) (a b : Math.Input K)
    (g : Nat → ℝ) (terminal : Math.Vec K) (T n : Nat)
    (hq : a.q=b.q) (hk : a.k=b.k) (hb : a.beta=b.beta) :
    Math.reverse s a g terminal T n=Math.reverse s b g terminal T n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    funext i
    simp only [Math.reverse,Math.pullback,hq,hk,hb,ih]

#axiomsClean reverse_h_math
#axiomsClean raw_dv_math
#axiomsClean adjoint_primal_independent
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Real
