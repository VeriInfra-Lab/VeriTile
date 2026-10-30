import bench.tritonbench_g.chunk_linear_attn.RealVJP
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def before {T K V : Nat} (x : Inputs T K V) (BT c : Nat) (e : Fin K) (j : Fin V) : ℝ :=
  x.initial e j+∑ u : Fin T, if u.val/BT<c then x.k u e*x.v u j else 0
def after {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (BT c : Nat) (e : Fin K) (j : Fin V) : ℝ :=
  R e j+scale*∑ t : Fin T, if c<t.val/BT then x.q t e*G t j else 0

theorem causal_partition (BT u t : Nat) (a : ℝ) :
    (if u/BT<t/BT then a else 0)+(if u/BT=t/BT ∧ u≤t then a else 0)=(if u≤t then a else 0) := by
  by_cases h : u≤t
  · have hd : u/BT≤t/BT := Nat.div_le_div_right h
    by_cases he : u/BT=t/BT
    · simp [h,he]
    · have hl : u/BT<t/BT := by omega
      simp [h,he,hl]
  · have hd : t/BT≤u/BT := Nat.div_le_div_right (show t≤u from by omega)
    have hn : ¬u/BT<t/BT := by omega
    simp [h,hn]

theorem state_partition {T K V : Nat} (x : Inputs T K V) (BT : Nat) (t : Fin T) (e : Fin K) (j : Fin V) :
    state x t e j=before x BT (t.val/BT) e j+
      ∑ u : Fin T, if u.val/BT=t.val/BT ∧ u≤t then x.k u e*x.v u j else 0 := by
  unfold state before
  rw [add_assoc,← Finset.sum_add_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro u _
  exact (causal_partition BT u.val t.val _).symm

theorem adjoint_partition {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (BT : Nat) (u : Fin T) (e : Fin K) (j : Fin V) :
    adjoint scale x G R u e j=after scale x G R BT (u.val/BT) e j+
      scale*∑ t : Fin T, if u.val/BT=t.val/BT ∧ u≤t then x.q t e*G t j else 0 := by
  unfold adjoint after
  rw [add_assoc,← mul_add,← Finset.sum_add_distrib]
  congr 2
  apply Finset.sum_congr rfl
  intro t _
  exact (causal_partition BT u.val t.val _).symm

def chunkOutput {T K V : Nat} (scale : ℝ) (x : Inputs T K V) (BT : Nat) (t : Fin T) (j : Fin V) : ℝ :=
  ((∑ e : Fin K, x.q t e*before x BT (t.val/BT) e j)+
    ∑ u : Fin T, if u.val/BT=t.val/BT ∧ u≤t then (∑ e : Fin K, x.q t e*x.k u e)*x.v u j else 0)*scale

def chunkGradients {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (BT : Nat) : Inputs T K V where
  q := fun t e => (∑ j : Fin V, G t j*before x BT (t.val/BT) e j)*scale+
    ∑ u : Fin T, if u.val/BT=t.val/BT ∧ u≤t then (∑ j : Fin V, G t j*x.v u j)*scale*x.k u e else 0
  k := fun u e => (∑ j : Fin V, x.v u j*after scale x G R BT (u.val/BT) e j)+
    ∑ t : Fin T, if u.val/BT=t.val/BT ∧ u≤t then x.q t e*((∑ j : Fin V, G t j*x.v u j)*scale) else 0
  v := fun u j => (∑ e : Fin K, x.k u e*after scale x G R BT (u.val/BT) e j)+
    ∑ t : Fin T, if u.val/BT=t.val/BT ∧ u≤t then ((∑ e : Fin K, x.k u e*x.q t e)*scale)*G t j else 0
  initial := fun e j => R e j+∑ t : Fin T, (x.q t e*scale)*G t j

theorem chunkOutput_correct {T K V : Nat} (scale : ℝ) (x : Inputs T K V) (BT : Nat) (t : Fin T) (j : Fin V) :
    chunkOutput scale x BT t j=output scale x t j := by
  unfold chunkOutput output
  simp_rw [state_partition x BT t]
  simp [mul_add,Finset.sum_add_distrib,Finset.mul_sum,Finset.sum_mul,mul_ite,ite_mul,
    Finset.sum_ite_irrel,mul_comm,mul_left_comm,mul_assoc]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  by_cases h : u.val/BT=t.val/BT ∧ u≤t <;> simp [h]

theorem chunkGradients_correct {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (BT : Nat) :
    chunkGradients scale x G R BT=gradients scale x G R := by
  unfold chunkGradients gradients
  congr 1
  · funext t e
    simp_rw [state_partition x BT t]
    simp [mul_add,add_mul,Finset.sum_add_distrib,Finset.mul_sum,Finset.sum_mul,mul_ite,ite_mul,
      Finset.sum_ite_irrel,mul_comm,mul_left_comm,mul_assoc]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro a _
    split_ifs <;> simp_all
  · funext u e
    simp_rw [adjoint_partition scale x G R BT u]
    simp [mul_add,add_mul,Finset.sum_add_distrib,Finset.mul_sum,Finset.sum_mul,mul_ite,ite_mul,
      Finset.sum_ite_irrel,mul_comm,mul_left_comm,mul_assoc]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro a _
    split_ifs <;> simp_all
  · funext u j
    simp_rw [adjoint_partition scale x G R BT u]
    simp [mul_add,add_mul,Finset.sum_add_distrib,Finset.mul_sum,Finset.sum_mul,mul_ite,ite_mul,
      Finset.sum_ite_irrel,mul_comm,mul_left_comm,mul_assoc]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro a _
    split_ifs <;> simp_all
  · funext e j
    simp [chunkGradients,gradients,Finset.mul_sum,mul_comm,mul_left_comm,mul_assoc]

theorem chunkGradients_vjp {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (BT : Nat) :
    IsVJP scale x G R (chunkGradients scale x G R BT) := by
  rw [chunkGradients_correct]
  exact gradients_correct scale x G R

#axiomsClean state_partition
#axiomsClean adjoint_partition
#axiomsClean chunkOutput_correct
#axiomsClean chunkGradients_correct
#axiomsClean chunkGradients_vjp
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
