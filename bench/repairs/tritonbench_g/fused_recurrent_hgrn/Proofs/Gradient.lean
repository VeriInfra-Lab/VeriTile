import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Whole
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def tangent (input : Input ℝ) (U : Bool) (bh d : Nat) (vx vg : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => input "g" bh n d*tangent input U bh d vx vg n +
      state realOps U input bh d n*vg n + vx n

def pathState (input : Input ℝ) (U : Bool) (bh d : Nat) (vx vg : Nat → ℝ) (e : ℝ) : Nat → ℝ
  | 0 => seed realOps U input bh d
  | n+1 => (input "g" bh n d+e*vg n)*pathState input U bh d vx vg e n+
      (input "x" bh n d+e*vx n)

theorem pathState_zero (input : Input ℝ) (U : Bool) (bh d : Nat) (vx vg : Nat → ℝ) (n : Nat) :
    pathState input U bh d vx vg 0 n=state realOps U input bh d n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [pathState,state,realOps,ih]

theorem pathState_derivative (input : Input ℝ) (U : Bool) (bh d : Nat) (vx vg : Nat → ℝ) (n : Nat) :
    HasDerivAt (fun e => pathState input U bh d vx vg e n) (tangent input U bh d vx vg n) 0 := by
  induction n with
  | zero => simpa [pathState,tangent] using hasDerivAt_const (0 : ℝ) (seed realOps U input bh d)
  | succ n ih =>
    have hg := (hasDerivAt_const (0 : ℝ) (input "g" bh n d)).add ((hasDerivAt_id (0 : ℝ)).mul_const (vg n))
    have hx := (hasDerivAt_const (0 : ℝ) (input "x" bh n d)).add ((hasDerivAt_id (0 : ℝ)).mul_const (vx n))
    have h := (hg.mul ih).add hx
    convert h using 1 <;> simp [pathState,tangent,pathState_zero] <;> ring

theorem previous_real (input : Input ℝ) (U : Bool) (bh d t : Nat) :
    previous realOps U input bh d t=state realOps U input bh d t := by
  by_cases ht : 0 < t
  · simp [previous,ht]
  · have h : t=0 := by omega
    subst t
    cases U <;> simp [previous,state,seed,realOps]

def boundary (input : Input ℝ) (V : Bool) (T bh d t : Nat) := adjoint realOps V input T bh d (T-t)

theorem boundary_step (input : Input ℝ) (V : Bool) (T bh d t : Nat) (ht : t < T) :
    boundary input V T bh d t=dx realOps V input T bh d t*input "g" bh t d ∧
    dx realOps V input T bh d t=boundary input V T bh d (t+1)+input "do" bh t d := by
  have a : T-t=(T-1-t)+1 := by omega
  have b : T-1-(T-1-t)=t := by omega
  have c : T-(t+1)=T-1-t := by omega
  constructor
  · simp [boundary,a,adjoint,dx,realOps,b]
  · simp [boundary,c,dx,realOps]

theorem gradient_pairing (input : Input ℝ) (U V : Bool) (T bh d : Nat) (vx vg : Nat → ℝ) :
    (∑ t ∈ Finset.range T, (dx realOps V input T bh d t*vx t + dg realOps U V input T bh d t*vg t))=
      (∑ t ∈ Finset.range T, input "do" bh t d*tangent input U bh d vx vg (t+1))+
      terminal realOps V input bh d*tangent input U bh d vx vg T := by
  have accumulated : ∀ n, n≤T →
      (∑ t ∈ Finset.range n, (dx realOps V input T bh d t*vx t + dg realOps U V input T bh d t*vg t))=
        (∑ t ∈ Finset.range n, input "do" bh t d*tangent input U bh d vx vg (t+1))+
        boundary input V T bh d n*tangent input U bh d vx vg n := by
    intro n
    induction n with
    | zero => intro _; simp [tangent]
    | succ n ih =>
      intro hn
      have hlt : n < T := by omega
      obtain ⟨hprev,hnext⟩ := boundary_step input V T bh d n hlt
      rw [Finset.sum_range_succ,Finset.sum_range_succ,ih (by omega)]
      have hdg : dg realOps U V input T bh d n=
          dx realOps V input T bh d n*state realOps U input bh d n := by
        change dx realOps V input T bh d n*previous realOps U input bh d n=_
        rw [previous_real]
      have htan : tangent input U bh d vx vg (n+1)=input "g" bh n d*tangent input U bh d vx vg n+
          state realOps U input bh d n*vg n+vx n := rfl
      rw [hdg,hprev]
      simp only [hnext,htan]
      ring
  simpa [boundary,adjoint] using accumulated T (by omega)

/-- The directional derivative of the scalar loss, with h0 held detached, is
the pairing with the dx/dg values computed by the repaired source. -/
theorem loss_derivative (input : Input ℝ) (U V : Bool) (T bh d : Nat) (vx vg : Nat → ℝ) :
    HasDerivAt (fun e =>
      (∑ t ∈ Finset.range T, input "do" bh t d*pathState input U bh d vx vg e (t+1))+
        terminal realOps V input bh d*pathState input U bh d vx vg e T)
      (∑ t ∈ Finset.range T, (dx realOps V input T bh d t*vx t+dg realOps U V input T bh d t*vg t)) 0 := by
  rw [gradient_pairing]
  apply HasDerivAt.add
  · have hs := HasDerivAt.sum (u:=Finset.range T)
      (fun t _ => (pathState_derivative input U bh d vx vg (t+1)).const_mul (input "do" bh t d))
    convert hs using 1
    funext e
    simp only [Finset.sum_apply]
  · exact (pathState_derivative input U bh d vx vg T).const_mul _

#axiomsClean loss_derivative
#specNonCircular pathState avoiding [FP.Structural.exec,Execution.gridRun]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
