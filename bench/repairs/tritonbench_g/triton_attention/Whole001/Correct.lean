import bench.repairs.tritonbench_g.triton_attention.Whole001.FPEquiv
import bench.repairs.tritonbench_g.triton_attention.Whole001.Kernels
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.MathDerivative

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001
namespace RealProgram
open Memory
open scoped BigOperators
noncomputable section

/-- No saved-result premise: both DO/L and Delta are obtained by reading
what the preceding forward stage actually stored. -/
theorem ds_connection (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups)
    (i j : Fin c.rows) :
    ds c (preprocess c (forward c s scale)) scale h i j =
      Math.sourceDS (readTensor c s .q h) (readTensor c s .k h)
        (readTensor c s .v h) (readTensor c s .g h) (forwardY c s scale h) scale
        (forwardL c s scale h) (forwardM c s scale h) i j := by
  simp [ds,probability,preprocess,forward,newDO,delta,Math.sourceDS]

theorem dq_connection (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :
    dq c (preprocess c (forward c s scale)) scale h =
      Math.tiledDQ c (readTensor c s .q h) (readTensor c s .k h)
        (readTensor c s .v h) (readTensor c s .g h) scale := by
  funext i e
  simp only [dq,ds_connection]
  simp [preprocess,forward,Math.tiledDQ,forwardY,forwardL,forwardM]

theorem dk_connection (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :
    dk c (preprocess c (forward c s scale)) scale h =
      Math.tiledDK c (readTensor c s .q h) (readTensor c s .k h)
        (readTensor c s .v h) (readTensor c s .g h) scale := by
  funext j e
  simp only [dk,ds_connection]
  simp [preprocess,forward,Math.tiledDK,forwardY,forwardL,forwardM]

theorem dv_connection (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :
    dv c (preprocess c (forward c s scale)) scale h =
      Math.tiledDV c (readTensor c s .q h) (readTensor c s .k h)
        (readTensor c s .v h) (readTensor c s .g h) scale := by
  funext j e
  simp [dv,probability,preprocess,forward,newDO,Math.tiledDV,forwardL,forwardM]

/-- Complete logical-memory output readbacks for the erased-real model.
Every batch/head and row/channel is covered; initial scratch memory is arbitrary.
This theorem intentionally makes no claim about native floating-point values. -/
theorem whole_memory_values (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :
    readTensor c (run c s scale) .o h =
      Math.attention (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h) scale ∧
    readTensor c (run c s scale) .dq h =
      Math.gradientQ (readTensor c s .q h) (readTensor c s .k h)
        (readTensor c s .v h) (readTensor c s .g h) scale ∧
    readTensor c (run c s scale) .dk h =
      Math.gradientK (readTensor c s .q h) (readTensor c s .k h)
        (readTensor c s .v h) (readTensor c s .g h) scale ∧
    readTensor c (run c s scale) .dv h =
      Math.gradientV (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .g h) scale := by
  have hv := Math.whole_logical_tensors c (readTensor c s .q h) (readTensor c s .k h)
    (readTensor c s .v h) (readTensor c s .g h) scale
  refine ⟨?_,?_,?_,?_⟩
  · simpa [run,backward,preprocess,forward,forwardY] using hv.1
  · simpa [run,backward,dq_connection] using hv.2.1
  · simpa [run,backward,dk_connection] using hv.2.2.1
  · simpa [run,backward,dv_connection] using hv.2.2.2

/-- The actual erased-real backward memory is the VJP of the independent
attention function, for every simultaneous Q/K/V perturbation. -/
theorem whole_memory_vjp (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups)
    (dQ dK dV : Math.Tensor c.rows c.dim) :
    HasDerivAt (fun t => Math.loss
      (Math.perturb (readTensor c s .q h) dQ t)
      (Math.perturb (readTensor c s .k h) dK t)
      (Math.perturb (readTensor c s .v h) dV t) (readTensor c s .g h) scale)
      ((∑ i, ∑ e, readTensor c (run c s scale) .dq h i e * dQ i e) +
       (∑ j, ∑ e, readTensor c (run c s scale) .dk h j e * dK j e) +
        ∑ j, ∑ e, readTensor c (run c s scale) .dv h j e * dV j e) 0 := by
  have hv := whole_memory_values c s scale h
  rw [hv.2.1,hv.2.2.1,hv.2.2.2]
  exact Math.attention_vjp _ _ _ _ _ _ _ _

def Written (c : Shape) (r : Buffer) (o : Nat) : Prop :=
  ((r = .o ∨ r = .newDO ∨ r = .dq ∨ r = .dk ∨ r = .dv) ∧ o < c.elements) ∨
  ((r = .l ∨ r = .m ∨ r = .delta) ∧ o < c.groups * c.rows)

private theorem tensor_frame (c : Shape) (s : Memory ℝ) (b : Buffer)
    (x : Coord c → ℝ) (r : Buffer) (o : Nat) (hn : ¬Written c r o)
    (hb : b = .o ∨ b = .newDO ∨ b = .dq ∨ b = .dk ∨ b = .dv) :
    writeTensor c s b x r o = s r o := by
  by_cases hr : r = b
  · apply writeTensor_outside
    by_contra! ho
    apply hn
    exact Or.inl ⟨hr ▸ hb,ho⟩
  · exact writeTensor_other c s b x r o hr

private theorem rows_frame (c : Shape) (s : Memory ℝ) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → ℝ) (r : Buffer) (o : Nat) (hn : ¬Written c r o)
    (hb : b = .l ∨ b = .m ∨ b = .delta) :
    writeRows c s b x r o = s r o := by
  by_cases hr : r = b
  · apply writeRows_outside
    by_contra! ho
    apply hn
    exact Or.inr ⟨hr ▸ hb,ho⟩
  · exact writeRows_other c s b x r o hr

/-- Exact full-memory frame, including every cell outside tensor extents. -/
theorem whole_memory_frame (c : Shape) (s : Memory ℝ) (scale : ℝ) (r : Buffer) (o : Nat)
    (hn : ¬Written c r o) : run c s scale r o = s r o := by
  simp only [run,backward,preprocess,forward]
  repeat' first
    | rw [tensor_frame c _ _ _ r o hn (by simp)]
    | rw [rows_frame c _ _ _ r o hn (by simp)]

/-- All input words/values retain their current contents. -/
theorem whole_memory_inputs (c : Shape) (s : Memory ℝ) (scale : ℝ) (o : Nat) :
    run c s scale .q o = s .q o ∧ run c s scale .k o = s .k o ∧
    run c s scale .v o = s .v o ∧ run c s scale .g o = s .g o := by
  refine ⟨?_,?_,?_,?_⟩ <;> apply whole_memory_frame <;> simp [Written]

#axiomsClean ds_connection
#axiomsClean whole_memory_values
#axiomsClean whole_memory_vjp
#axiomsClean whole_memory_frame
#axiomsClean whole_memory_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end RealProgram
end VeriTile.Bench.Repairs.TritonAttention.Whole001
