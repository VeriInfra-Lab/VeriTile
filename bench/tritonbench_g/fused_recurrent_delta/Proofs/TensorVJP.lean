import bench.tritonbench_g.fused_recurrent_delta.Proofs.Mathematics
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Math
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 5000000
structure TensorInput (BH K V : Nat) where
  q : Fin BH → Nat → Vec K
  k : Fin BH → Nat → Vec K
  x : Fin BH → Nat → Fin V → ℝ
  beta : Fin BH → Nat → Fin V → ℝ
  h0 : Fin BH → Fin V → Vec K
def column {BH K V : Nat} (a : TensorInput BH K V) (bh : Fin BH) (v : Fin V) : Input K :=
  ⟨a.q bh,a.k bh,fun t => a.x bh t v,fun t => a.beta bh t v,a.h0 bh v⟩
def tensorLoss {BH K V : Nat} (s : ℝ) (a : TensorInput BH K V)
    (g : Fin BH → Nat → Fin V → ℝ) (terminal : Fin BH → Fin V → Vec K) (T : Nat) :=
  ∑ bh,∑ v,loss s (column a bh v) (fun t => g bh t v) (terminal bh v) T
def tensorPerturb {BH K V : Nat} (a d : TensorInput BH K V) (e : ℝ) : TensorInput BH K V :=
  ⟨fun b t k => a.q b t k+e*d.q b t k,
   fun b t k => a.k b t k+e*d.k b t k,
   fun b t v => a.x b t v+e*d.x b t v,
   fun b t v => a.beta b t v+e*d.beta b t v,
   fun b v k => a.h0 b v k+e*d.h0 b v k⟩
theorem column_perturb {BH K V : Nat} (a d : TensorInput BH K V) (e : ℝ) (bh : Fin BH) (v : Fin V) :
    column (tensorPerturb a d e) bh v=perturb (column a bh v) (column d bh v) e := rfl
def tensorGradient {BH K V : Nat} (s : ℝ) (a : TensorInput BH K V)
    (g : Fin BH → Nat → Fin V → ℝ) (terminal : Fin BH → Fin V → Vec K) (T : Nat) :
    TensorInput BH K V :=
  let z:=fun bh v t => gradient s (column a bh v) (fun n => g bh n v) (terminal bh v) T t
  ⟨fun bh t k => ∑ v,(z bh v t).q k,
   fun bh t k => ∑ v,(z bh v t).k k,
   fun bh t v => (z bh v t).x,
   fun bh t v => (z bh v t).beta,
   fun bh v => carry s (column a bh v) (fun n => g bh n v) (terminal bh v) T 0⟩
def tensorPair {BH K V : Nat} (z d : TensorInput BH K V) (T : Nat) :=
  ∑ bh,((∑ v,dot (z.h0 bh v) (d.h0 bh v))+
    ∑ t ∈ Finset.range T,(dot (z.q bh t) (d.q bh t)+dot (z.k bh t) (d.k bh t)+
      ∑ v,(z.x bh t v*d.x bh t v+z.beta bh t v*d.beta bh t v)))

theorem tensor_pairing {BH K V : Nat} (s : ℝ) (a d : TensorInput BH K V)
    (g : Fin BH → Nat → Fin V → ℝ) (terminal : Fin BH → Fin V → Vec K) (T : Nat) :
    (∑ bh,∑ v,(dot (carry s (column a bh v) (fun n => g bh n v) (terminal bh v) T 0) (d.h0 bh v)+
      ∑ t ∈ Finset.range T,inputPair (gradient s (column a bh v) (fun n => g bh n v) (terminal bh v) T t) (column d bh v) t))=
      tensorPair (tensorGradient s a g terminal T) d T := by
  unfold tensorPair tensorGradient
  apply Finset.sum_congr rfl
  intro bh _
  rw [Finset.sum_add_distrib]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  simp only [inputPair,column,←add_assoc,Finset.sum_add_distrib,shared_input_pairing]

/-- Full BH-by-T-by-K/V VJP, after aggregating the shared q/k coordinates. -/
theorem tensor_loss_derivative {BH K V : Nat} (s : ℝ) (a d : TensorInput BH K V)
    (g : Fin BH → Nat → Fin V → ℝ) (terminal : Fin BH → Fin V → Vec K) (T : Nat) :
    HasDerivAt (fun e => tensorLoss s (tensorPerturb a d e) g terminal T)
      (tensorPair (tensorGradient s a g terminal T) d T) 0 := by
  rw [←tensor_pairing]
  exact HasDerivAt.fun_sum (fun bh _ => HasDerivAt.fun_sum (fun v _ =>
    loss_derivative s (column a bh v) (column d bh v) (fun n => g bh n v) (terminal bh v) T))

/-- A shared per-token beta collects its column derivatives. Choosing a zero
H0 perturbation gives detached/absent H0; zero terminal gives absent dHT. -/
theorem scalar_beta_vjp {BH V : Nat} (z : Fin BH → Nat → Fin V → ℝ)
    (d : Fin BH → Nat → ℝ) (T : Nat) :
    (∑ bh,∑ t ∈ Finset.range T,∑ v,z bh t v*d bh t)=
      ∑ bh,∑ t ∈ Finset.range T,(∑ v,z bh t v)*d bh t := by
  simp only [Finset.sum_mul]

#axiomsClean tensor_loss_derivative
#axiomsClean scalar_beta_vjp
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Math
