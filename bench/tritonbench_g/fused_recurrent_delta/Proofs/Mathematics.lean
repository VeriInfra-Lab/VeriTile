import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
/-! Independent delta recurrence and its complete VJP. A column is a value
feature, not a source tile. q/k are shared across columns; scalar beta shares
beta across columns. These definitions mention no kernel, memory or schedule. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Math
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
abbrev Vec (K : Nat) := Fin K → ℝ
def dot {K : Nat} (a b : Vec K) : ℝ := ∑ i,a i*b i
theorem dot_comm {K : Nat} (a b : Vec K) : dot a b=dot b a := by
  unfold dot
  apply Finset.sum_congr rfl
  intro i _
  ring
theorem dot_add_left {K : Nat} (a b c : Vec K) :
    dot (fun i => a i+b i) c=dot a c+dot b c := by simp [dot,add_mul,Finset.sum_add_distrib]
theorem dot_sub_left {K : Nat} (a b c : Vec K) :
    dot (fun i => a i-b i) c=dot a c-dot b c := by simp [dot,sub_mul,Finset.sum_sub_distrib]
theorem dot_mul_left {K : Nat} (r : ℝ) (a b : Vec K) :
    dot (fun i => r*a i) b=r*dot a b := by simp [dot,mul_assoc,Finset.mul_sum]
theorem dot_add_right {K : Nat} (a b c : Vec K) :
    dot a (fun i => b i+c i)=dot a b+dot a c := by simp [dot,mul_add,Finset.sum_add_distrib]
theorem dot_mul_right {K : Nat} (r : ℝ) (a b : Vec K) :
    dot a (fun i => r*b i)=r*dot a b := by
  rw [dot_comm,dot_mul_left,dot_comm b a]

structure Input (K : Nat) where
  q : Nat → Vec K
  k : Nat → Vec K
  x : Nat → ℝ
  beta : Nat → ℝ
  h0 : Vec K

def residual {K : Nat} (h k : Vec K) (x : ℝ) := x-dot h k
def next {K : Nat} (h k : Vec K) (x beta : ℝ) : Vec K :=
  fun i => h i+k i*(beta*residual h k x)
def state {K : Nat} (a : Input K) : Nat → Vec K
  | 0 => a.h0
  | t+1 => next (state a t) (a.k t) (a.x t) (a.beta t)
def output {K : Nat} (s : ℝ) (a : Input K) (t : Nat) := s*dot (state a (t+1)) (a.q t)
def loss {K : Nat} (s : ℝ) (a : Input K) (g : Nat → ℝ) (terminal : Vec K) (T : Nat) :=
  (∑ t ∈ Finset.range T,g t*output s a t)+dot terminal (state a T)

def tangentStep {K : Nat} (h k dh dk : Vec K) (x beta dx db : ℝ) : Vec K :=
  fun i => dh i+dk i*(beta*residual h k x)+
    k i*(db*residual h k x+beta*(dx-dot dh k-dot h dk))
def tangent {K : Nat} (a d : Input K) : Nat → Vec K
  | 0 => d.h0
  | t+1 => tangentStep (state a t) (a.k t) (tangent a d t) (d.k t)
      (a.x t) (a.beta t) (d.x t) (d.beta t)
def outputTangent {K : Nat} (s : ℝ) (a d : Input K) (t : Nat) :=
  s*(dot (tangent a d (t+1)) (a.q t)+dot (state a (t+1)) (d.q t))

structure Grad (K : Nat) where
  q : Vec K
  k : Vec K
  x : ℝ
  beta : ℝ
  h : Vec K
def pullback {K : Nat} (s : ℝ) (h q k : Vec K) (x beta g : ℝ) (G : Vec K) : Grad K :=
  let r:=residual h k x
  let u:=beta*r
  let E:=fun i => G i+s*g*q i
  let z:=dot E k
  {q:=fun i => s*g*next h k x beta i,
   k:=fun i => u*E i-(beta*z)*h i,
   x:=beta*z,beta:=r*z,h:=fun i => E i-(beta*z)*k i}

theorem step_adjoint {K : Nat} (s : ℝ) (h q k dh dq dk G : Vec K) (x beta dx db g : ℝ) :
    dot G (tangentStep h k dh dk x beta dx db)+
      g*(s*(dot (tangentStep h k dh dk x beta dx db) q+dot (next h k x beta) dq))=
    dot (pullback s h q k x beta g G).h dh+
      dot (pullback s h q k x beta g G).q dq+
      dot (pullback s h q k x beta g G).k dk+
      (pullback s h q k x beta g G).x*dx+(pullback s h q k x beta g G).beta*db := by
  let E:=fun i => G i+s*g*q i
  have pair (v : Vec K) : dot G v+g*s*dot v q=dot E v := by
    dsimp [E]
    rw [dot_add_left,dot_mul_left,dot_comm q v]
    ring
  have expand : tangentStep h k dh dk x beta dx db=
      fun i => dh i+(beta*residual h k x)*dk i+
        (db*residual h k x+beta*(dx-dot dh k-dot h dk))*k i := by
    funext i
    unfold tangentStep
    ring
  calc
    _ = dot E (tangentStep h k dh dk x beta dx db)+g*s*dot (next h k x beta) dq := by
      rw [←pair]
      ring
    _ = _ := by
      rw [expand,dot_add_right,dot_add_right,dot_mul_right,dot_mul_right]
      dsimp [pullback]
      simp only [dot_sub_left,dot_add_left,dot_mul_left]
      dsimp [E]
      simp only [dot_add_left,dot_mul_left]
      rw [dot_comm k dh]
      ring

def reverse {K : Nat} (s : ℝ) (a : Input K) (g : Nat → ℝ) (terminal : Vec K) (T : Nat) : Nat → Vec K
  | 0 => terminal
  | n+1 => (pullback s (state a (T-1-n)) (a.q (T-1-n)) (a.k (T-1-n))
      (a.x (T-1-n)) (a.beta (T-1-n)) (g (T-1-n)) (reverse s a g terminal T n)).h
def carry {K : Nat} (s : ℝ) (a : Input K) (g : Nat → ℝ) (terminal : Vec K) (T t : Nat) :=
  reverse s a g terminal T (T-t)
def gradient {K : Nat} (s : ℝ) (a : Input K) (g : Nat → ℝ) (terminal : Vec K) (T t : Nat) :=
  pullback s (state a t) (a.q t) (a.k t) (a.x t) (a.beta t) (g t)
    (carry s a g terminal T (t+1))
def inputPair {K : Nat} (z : Grad K) (d : Input K) (t : Nat) :=
  dot z.q (d.q t)+dot z.k (d.k t)+z.x*d.x t+z.beta*d.beta t

theorem carry_step {K : Nat} (s : ℝ) (a : Input K) (g : Nat → ℝ) (terminal : Vec K)
    (T t : Nat) (ht : t<T) : (gradient s a g terminal T t).h=carry s a g terminal T t := by
  have hn : T-t=(T-(t+1))+1 := by omega
  have hi : T-1-(T-(t+1))=t := by omega
  simp only [carry,hn,reverse,hi,gradient]

theorem whole_adjoint {K : Nat} (s : ℝ) (a d : Input K) (g : Nat → ℝ)
    (terminal : Vec K) (T : Nat) :
    (∑ t ∈ Finset.range T,g t*outputTangent s a d t)+dot terminal (tangent a d T)=
    dot (carry s a g terminal T 0) d.h0+
      ∑ t ∈ Finset.range T,inputPair (gradient s a g terminal T t) d t := by
  have accumulated : ∀ n,n≤T →
      (∑ t ∈ Finset.range n,g t*outputTangent s a d t)+
        dot (carry s a g terminal T n) (tangent a d n)=
      dot (carry s a g terminal T 0) d.h0+
        ∑ t ∈ Finset.range n,inputPair (gradient s a g terminal T t) d t := by
    intro n
    induction n with
    | zero => intro _; simp [tangent]
    | succ n ih =>
      intro hn
      have hprev:=carry_step s a g terminal T n (by omega)
      have hstep:=step_adjoint s (state a n) (a.q n) (a.k n) (tangent a d n)
        (d.q n) (d.k n) (carry s a g terminal T (n+1)) (a.x n) (a.beta n)
        (d.x n) (d.beta n) (g n)
      change dot (carry s a g terminal T (n+1)) (tangent a d (n+1))+
        g n*outputTangent s a d n=dot (gradient s a g terminal T n).h (tangent a d n)+
          dot (gradient s a g terminal T n).q (d.q n)+
          dot (gradient s a g terminal T n).k (d.k n)+
          (gradient s a g terminal T n).x*d.x n+(gradient s a g terminal T n).beta*d.beta n at hstep
      rw [hprev] at hstep
      rw [Finset.sum_range_succ,Finset.sum_range_succ]
      dsimp [inputPair]
      have old:=ih (by omega)
      dsimp [inputPair] at old
      linarith
  simpa [carry,reverse] using accumulated T le_rfl

def perturb {K : Nat} (a d : Input K) (e : ℝ) : Input K :=
  {q:=fun t i => a.q t i+e*d.q t i,k:=fun t i => a.k t i+e*d.k t i,
   x:=fun t => a.x t+e*d.x t,beta:=fun t => a.beta t+e*d.beta t,
   h0:=fun i => a.h0 i+e*d.h0 i}
@[simp] theorem perturb_zero {K : Nat} (a d : Input K) : perturb a d 0=a := by
  cases a
  simp [perturb]

theorem affine_derivative (a d : ℝ) : HasDerivAt (fun e => a+e*d) d 0 := by
  simpa using ((hasDerivAt_id (0:ℝ)).mul_const d).const_add a

theorem state_derivative {K : Nat} (a d : Input K) (n : Nat) (i : Fin K) :
    HasDerivAt (fun e => state (perturb a d e) n i) (tangent a d n i) 0 := by
  induction n generalizing i with
  | zero => exact affine_derivative _ _
  | succ n ih =>
    have hsum := HasDerivAt.fun_sum (u:=Finset.univ) (fun j _ =>
      (ih j).mul (affine_derivative (a.k n j) (d.k n j)))
    have hr := (affine_derivative (a.x n) (d.x n)).sub hsum
    have hu := (affine_derivative (a.beta n) (d.beta n)).mul hr
    have hn := (ih i).add ((affine_derivative (a.k n i) (d.k n i)).mul hu)
    convert hn using 1
    simp only [Pi.mul_apply,Pi.sub_apply,perturb_zero,zero_mul,add_zero]
    simp only [tangent,tangentStep,residual,dot,Finset.sum_add_distrib]
    ring

theorem output_derivative {K : Nat} (s : ℝ) (a d : Input K) (t : Nat) :
    HasDerivAt (fun e => output s (perturb a d e) t) (outputTangent s a d t) 0 := by
  have h:=(HasDerivAt.fun_sum (u:=Finset.univ) (fun i _ =>
    (state_derivative a d (t+1) i).mul (affine_derivative (a.q t i) (d.q t i)))).const_mul s
  convert h using 1
  simp only [perturb_zero,zero_mul,add_zero,outputTangent,dot,Finset.sum_add_distrib]

/-- Actual directional derivative, including q,k,x,beta and the initial state.
The terminal cotangent is arbitrary and may be zero when HT is not returned. -/
theorem loss_derivative {K : Nat} (s : ℝ) (a d : Input K) (g : Nat → ℝ)
    (terminal : Vec K) (T : Nat) :
    HasDerivAt (fun e => loss s (perturb a d e) g terminal T)
      (dot (carry s a g terminal T 0) d.h0+
        ∑ t ∈ Finset.range T,inputPair (gradient s a g terminal T t) d t) 0 := by
  rw [←whole_adjoint]
  exact (HasDerivAt.fun_sum (fun t _ => (output_derivative s a d t).const_mul (g t))).add
    (HasDerivAt.fun_sum (fun i _ => (state_derivative a d T i).const_mul (terminal i)))

theorem shared_input_pairing {K V : Nat} (grad : Fin V → Vec K) (d : Vec K) :
    (∑ v,dot (grad v) d)=dot (fun i => ∑ v,grad v i) d := by
  simp only [dot,Finset.sum_mul]
  exact Finset.sum_comm
theorem scalar_beta_pairing {V : Nat} (grad : Fin V → ℝ) (d : ℝ) :
    (∑ v,grad v*d)=(∑ v,grad v)*d := by rw [Finset.sum_mul]

#axiomsClean step_adjoint
#axiomsClean whole_adjoint
#axiomsClean state_derivative
#axiomsClean loss_derivative
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Math
