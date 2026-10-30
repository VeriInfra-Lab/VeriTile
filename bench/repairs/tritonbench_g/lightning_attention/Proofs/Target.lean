import bench.repairs.tritonbench_g.lightning_attention.Proofs.LogicalTarget
import bench.repairs.tritonbench_g.chunk_linear_attn.RealVJP
namespace VeriTile.Bench.Repairs.LightningAttention
open scoped BigOperators
noncomputable section
namespace Target
abbrev Legacy := VeriTile.Bench.TritonBenchG.LightningAttention.LogicalTarget.Matrix
abbrev Inputs := VeriTile.Bench.Repairs.ChunkLinearAttn.Real.Inputs
open VeriTile.Bench.Repairs.ChunkLinearAttn.Real

def input {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v : Fin N → Fin E → ℝ) : Inputs N D E :=
  ⟨q,k,v,fun _ _ => 0⟩
def score {N D : Nat} (q k : Fin N → Fin D → ℝ) (t u : Fin N) : ℝ :=
  ∑ a : Fin D, q t a*k u a
def output {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v : Fin N → Fin E → ℝ)
    (t : Fin N) (c : Fin E) : ℝ :=
  ∑ u : Fin N, if u≤t then score q k t u*v u c else 0
def dq {N D E : Nat} (k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ)
    (t : Fin N) (a : Fin D) : ℝ :=
  ∑ u : Fin N, if u≤t then (∑ c : Fin E, g t c*v u c)*k u a else 0
def dk {N D E : Nat} (q : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ)
    (u : Fin N) (a : Fin D) : ℝ :=
  ∑ t : Fin N, if u≤t then q t a*(∑ c : Fin E, g t c*v u c) else 0
def dv {N D E : Nat} (q k : Fin N → Fin D → ℝ) (g : Fin N → Fin E → ℝ)
    (u : Fin N) (c : Fin E) : ℝ :=
  ∑ t : Fin N, if u≤t then score q k t u*g t c else 0

theorem vjp {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ) :
    IsVJP 1 (input q k v) g (fun _ _ => 0)
      (gradients 1 (input q k v) g (fun _ _ => 0)) :=
  gradients_correct 1 (input q k v) g (fun _ _ => 0)

theorem output_bridge {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v : Fin N → Fin E → ℝ)
    (t : Fin N) (c : Fin E) :
    ChunkLinearAttn.Real.output 1 (input q k v) t c = output q k v t c := by
  simp only [ChunkLinearAttn.Real.output, state, input, output, score, one_mul, zero_add,
    Finset.mul_sum, Finset.sum_mul, mul_ite, mul_zero, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  by_cases h : u≤t <;> simp [h, Finset.sum_ite_irrel, mul_assoc]

theorem query_bridge {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ)
    (t : Fin N) (a : Fin D) :
    (gradients 1 (input q k v) g (fun _ _ => 0)).q t a = dq k v g t a := by
  simp only [gradients, state, input, dq, one_mul, zero_add,
    Finset.sum_mul, Finset.mul_sum, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  by_cases h : u≤t <;> simp [h, mul_comm, mul_left_comm, mul_assoc]

theorem key_bridge {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ)
    (u : Fin N) (a : Fin D) :
    (gradients 1 (input q k v) g (fun _ _ => 0)).k u a = dk q v g u a := by
  simp only [gradients, adjoint, input, dk, one_mul, zero_add,
    Finset.sum_mul, Finset.mul_sum, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  by_cases h : u≤t <;> simp [h, mul_assoc]

theorem value_bridge {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ)
    (u : Fin N) (c : Fin E) :
    (gradients 1 (input q k v) g (fun _ _ => 0)).v u c = dv q k g u c := by
  simp only [gradients, adjoint, input, dv, score, one_mul, zero_add,
    Finset.sum_mul, Finset.mul_sum, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  by_cases h : u≤t <;> simp [h, mul_comm, mul_left_comm, mul_assoc]

theorem coordinate_vjp {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ) :
    (∀ t a, HasDerivAt (fun ε => loss 1 (perturb (input q k v) (basisQ t a) ε) g (fun _ _ => 0)) (dq k v g t a) 0) ∧
    (∀ u a, HasDerivAt (fun ε => loss 1 (perturb (input q k v) (basisK u a) ε) g (fun _ _ => 0)) (dk q v g u a) 0) ∧
    (∀ u c, HasDerivAt (fun ε => loss 1 (perturb (input q k v) (basisV u c) ε) g (fun _ _ => 0)) (dv q k g u c) 0) := by
  have h := vjp q k v g
  exact ⟨fun t a => by simpa only [query_bridge] using h.query t a,
    fun u a => by simpa only [key_bridge] using h.key u a,
    fun u c => by simpa only [value_bridge] using h.value u c⟩

#axiomsClean vjp
#axiomsClean output_bridge
#axiomsClean coordinate_vjp
#specNonCircular output avoiding [gradients]
end Target
end
end VeriTile.Bench.Repairs.LightningAttention
