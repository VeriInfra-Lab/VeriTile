import bench.tritonbench_g.lightning_attention.Proofs.Target
namespace VeriTile.Bench.Repairs.LightningAttention.Target
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
/-- The scalar loss contracted directly with this task's independent causal output. -/
def loss {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ) : ℝ :=
  ∑ t : Fin N, ∑ c : Fin E, g t c * output q k v t c
/-- A perturbation of exactly one caller tensor coordinate. -/
def bump {N F : Nat} (x : Fin N → Fin F → ℝ) (row : Fin N) (col : Fin F) (ε : ℝ) :=
  fun t a => x t a + ε * (if t=row then if a=col then 1 else 0 else 0)

theorem loss_bridge {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ) :
    ChunkLinearAttn.Real.loss 1 (input q k v) g (fun _ _ => 0) = loss q k v g := by
  simp [ChunkLinearAttn.Real.loss,loss,output_bridge]

theorem perturb_query {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v : Fin N → Fin E → ℝ)
    (t : Fin N) (a : Fin D) (ε : ℝ) :
    ChunkLinearAttn.Real.perturb (input q k v) (ChunkLinearAttn.Real.basisQ t a) ε =
      input (bump q t a ε) k v := by
  simp [ChunkLinearAttn.Real.perturb,ChunkLinearAttn.Real.basisQ,ChunkLinearAttn.Real.zeroInputs,input]
  funext u f
  simp [Target.bump]
theorem perturb_key {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v : Fin N → Fin E → ℝ)
    (t : Fin N) (a : Fin D) (ε : ℝ) :
    ChunkLinearAttn.Real.perturb (input q k v) (ChunkLinearAttn.Real.basisK t a) ε =
      input q (bump k t a ε) v := by
  simp [ChunkLinearAttn.Real.perturb,ChunkLinearAttn.Real.basisK,ChunkLinearAttn.Real.zeroInputs,input]
  funext u f
  simp [Target.bump]
theorem perturb_value {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v : Fin N → Fin E → ℝ)
    (t : Fin N) (c : Fin E) (ε : ℝ) :
    ChunkLinearAttn.Real.perturb (input q k v) (ChunkLinearAttn.Real.basisV t c) ε =
      input q k (bump v t c ε) := by
  simp [ChunkLinearAttn.Real.perturb,ChunkLinearAttn.Real.basisV,ChunkLinearAttn.Real.zeroInputs,input]
  funext u f
  simp [Target.bump]

theorem loss_coordinate_derivatives {N D E : Nat} (q k : Fin N → Fin D → ℝ) (v g : Fin N → Fin E → ℝ) :
    (∀ t a, HasDerivAt (fun ε => loss (bump q t a ε) k v g) (dq k v g t a) 0) ∧
    (∀ t a, HasDerivAt (fun ε => loss q (bump k t a ε) v g) (dk q v g t a) 0) ∧
    (∀ t c, HasDerivAt (fun ε => loss q k (bump v t c ε) g) (dv q k g t c) 0) := by
  simpa only [perturb_query,perturb_key,perturb_value,loss_bridge] using coordinate_vjp q k v g

#axiomsClean loss_bridge
#axiomsClean perturb_query
#axiomsClean perturb_key
#axiomsClean perturb_value
#axiomsClean loss_coordinate_derivatives
#specNonCircular loss avoiding [dq,dk,dv,ChunkLinearAttn.Real.gradients]
end
end VeriTile.Bench.Repairs.LightningAttention.Target
