import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Independent local chunk loss. Coefficients contain the fixed value,
history and cotangent contractions; query, key and cumulative gates vary.
No source program, interpreter, register, output buffer or native operation
is used to define this loss or its gradients. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelMath
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

structure Input (N E : Nat) where
  query : Fin N → Fin E → ℝ
  key : Fin N → Fin E → ℝ
  gate : Fin N → ℝ
structure Coefficients (N E : Nat) where
  incoming : Fin N → Fin E → ℝ
  outgoing : Fin N → Fin E → ℝ
  score : Fin N → Fin N → ℝ
  history : Fin E → ℝ

def loss (c : Coefficients N E) (a : Input N E) (last : Fin N) : ℝ :=
  (∑ r,∑ e,a.query r e*Real.exp (a.gate r)*c.incoming r e)+
  (∑ r,∑ u,∑ e,(a.query r e*a.key u e)*Real.exp (a.gate r-a.gate u)*c.score r u)+
  (∑ e,c.history e)*Real.exp (a.gate last)+
  (∑ r,∑ e,a.key r e*Real.exp (a.gate last-a.gate r)*c.outgoing r e)

def perturb (a d : Input N E) (x : ℝ) : Input N E :=
  ⟨fun r e=>a.query r e+x*d.query r e,fun r e=>a.key r e+x*d.key r e,fun r=>a.gate r+x*d.gate r⟩

def baseQ (c : Coefficients N E) (a : Input N E) (r : Fin N) (e : Fin E) : ℝ :=
  Real.exp (a.gate r)*c.incoming r e
def baseK (c : Coefficients N E) (a : Input N E) (last r : Fin N) (e : Fin E) : ℝ :=
  Real.exp (a.gate last-a.gate r)*c.outgoing r e
def score (c : Coefficients N E) (a : Input N E) (r u : Fin N) : ℝ :=
  Real.exp (a.gate r-a.gate u)*c.score r u
def dq (c : Coefficients N E) (a : Input N E) (r : Fin N) (e : Fin E) : ℝ :=
  baseQ c a r e+∑ u,score c a r u*a.key u e
def dk (c : Coefficients N E) (a : Input N E) (last r : Fin N) (e : Fin E) : ℝ :=
  baseK c a last r e+∑ u,score c a u r*a.query u e
def correction (c : Coefficients N E) (a : Input N E) (last : Fin N) : ℝ :=
  (∑ e,c.history e)*Real.exp (a.gate last)+∑ r,∑ e,baseK c a last r e*a.key r e
def dg (c : Coefficients N E) (a : Input N E) (last r : Fin N) : ℝ :=
  (∑ e,(a.query r e*dq c a r e-a.key r e*dk c a last r e))+
  if r=last then correction c a last else 0

def tangent (c : Coefficients N E) (a d : Input N E) (last : Fin N) : ℝ :=
  (∑ r,∑ e,(d.query r e*Real.exp (a.gate r)+a.query r e*(Real.exp (a.gate r)*d.gate r))*c.incoming r e)+
  (∑ r,∑ u,∑ e,((d.query r e*a.key u e+a.query r e*d.key u e)*Real.exp (a.gate r-a.gate u)+
    (a.query r e*a.key u e)*(Real.exp (a.gate r-a.gate u)*(d.gate r-d.gate u)))*c.score r u)+
  (∑ e,c.history e)*(Real.exp (a.gate last)*d.gate last)+
  (∑ r,∑ e,(d.key r e*Real.exp (a.gate last-a.gate r)+
    a.key r e*(Real.exp (a.gate last-a.gate r)*(d.gate last-d.gate r)))*c.outgoing r e)

theorem affine_derivative (a d : ℝ) : HasDerivAt (fun x=>a+x*d) d 0 := by
  simpa using ((hasDerivAt_id (0:ℝ)).mul_const d).const_add a

theorem loss_derivative (c : Coefficients N E) (a d : Input N E) (last : Fin N) :
    HasDerivAt (fun x=>loss c (perturb a d x) last) (tangent c a d last) 0 := by
  have hq r e:=affine_derivative (a.query r e) (d.query r e)
  have hk r e:=affine_derivative (a.key r e) (d.key r e)
  have hg r:=affine_derivative (a.gate r) (d.gate r)
  have incoming := HasDerivAt.fun_sum (u:=Finset.univ) (fun r _=>
    HasDerivAt.fun_sum (u:=Finset.univ) (fun e _=>((hq r e).mul (hg r).exp).mul_const (c.incoming r e)))
  have attention := HasDerivAt.fun_sum (u:=Finset.univ) (fun r _=>
    HasDerivAt.fun_sum (u:=Finset.univ) (fun u _=>
      HasDerivAt.fun_sum (u:=Finset.univ) (fun e _=>
        (((hq r e).mul (hk u e)).mul ((hg r).sub (hg u)).exp).mul_const (c.score r u))))
  have terminal := ((hg last).exp).const_mul (∑ e,c.history e)
  have outgoing := HasDerivAt.fun_sum (u:=Finset.univ) (fun r _=>
    HasDerivAt.fun_sum (u:=Finset.univ) (fun e _=>
      ((hk r e).mul ((hg last).sub (hg r)).exp).mul_const (c.outgoing r e)))
  simpa only [loss,perturb,tangent,Pi.add_apply,Pi.mul_apply,Pi.sub_apply,zero_mul,add_zero]
    using ((incoming.add attention).add terminal).add outgoing

def pairing (c : Coefficients N E) (a d : Input N E) (last : Fin N) : ℝ :=
  (∑ r,∑ e,dq c a r e*d.query r e)+
  (∑ r,∑ e,dk c a last r e*d.key r e)+
  ∑ r,dg c a last r*d.gate r

theorem tangent_pairing (c : Coefficients N E) (a d : Input N E) (last : Fin N) :
    tangent c a d last=pairing c a d last := by
  classical
  have swap (f : Fin N → Fin E → Fin N → ℝ) :
      (∑ r,∑ e,∑ u,f r e u) = ∑ r,∑ u,∑ e,f r e u := by
    apply Finset.sum_congr rfl
    intro r _
    exact Finset.sum_comm
  simp only [pairing,dq,dk,dg,baseQ,baseK,score,correction,
    Finset.sum_add_distrib,Finset.sum_sub_distrib,
    Finset.sum_mul,Finset.mul_sum,add_mul,sub_mul,mul_add,mul_sub,ite_mul,zero_mul,
    Finset.sum_ite_eq',Finset.mem_univ,if_true]
  simp only [tangent,add_mul,sub_mul,mul_add,mul_sub,
    Finset.sum_add_distrib,Finset.sum_sub_distrib,Finset.sum_mul,swap]
  rw [show (∑ r,∑ u,∑ e,Real.exp (a.gate u-a.gate r)*c.score u r*a.query u e*d.key r e)=
      ∑ u,∑ r,∑ e,Real.exp (a.gate u-a.gate r)*c.score u r*a.query u e*d.key r e from Finset.sum_comm]
  rw [show (∑ r,∑ u,∑ e,a.key r e*(Real.exp (a.gate u-a.gate r)*c.score u r*a.query u e)*d.gate r)=
      ∑ u,∑ r,∑ e,a.key r e*(Real.exp (a.gate u-a.gate r)*c.score u r*a.query u e)*d.gate r from Finset.sum_comm]
  simp only [mul_comm,mul_left_comm,mul_assoc]
  abel

theorem gradients (c : Coefficients N E) (a d : Input N E) (last : Fin N) :
    HasDerivAt (fun x=>loss c (perturb a d x) last) (pairing c a d last) 0 := by
  rw [← tangent_pairing]
  exact loss_derivative c a d last

#axiomsClean loss_derivative
#axiomsClean gradients
#specNonCircular loss avoiding [VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelMath.dq,
  VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelMath.dk,VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelMath.dg]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelMath
