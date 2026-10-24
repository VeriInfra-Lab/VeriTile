import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.ReaderCorrect

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Correct
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- Causal softmax over the complete logical sequence, independently of tiles,
saved maxima and the materialized source stages. -/
def probability {N D : Nat} (Q K : Math.Tensor N D) (scale : ℝ) (i j : Fin N) : ℝ :=
  (if j ≤ i then Real.exp ((∑ e, Q i e * K j e) * scale) else 0) /
    ∑ k, if k ≤ i then Real.exp ((∑ e, Q i e * K k e) * scale) else 0

/-- The ordinary softmax Jacobian pullback of DO * Vᵀ. -/
def scoreCotangent {N D : Nat} (Q K V G : Math.Tensor N D) (scale : ℝ) (i j : Fin N) : ℝ :=
  probability Q K scale i j * ((∑ e, G i e * V j e) -
    ∑ k, probability Q K scale i k * (∑ e, G i e * V k e))

/-- Logical channels are Out, DQ, DK and DV in that order. Every sum uses the
full logical sequence/head dimension; no saved source result is an input. -/
def attentionAndVJP (c : Shape) (scale : ℝ) (xs : PublicIO.LogicalInputs c) : PublicIO.LogicalOutputs c :=
  ⟨fun h i d => ∑ j, probability (xs.query h) (xs.key h) scale i j * xs.value h j d,
   fun h i d => ∑ j, (scoreCotangent (xs.query h) (xs.key h) (xs.value h)
     (xs.outputCotangent h) scale i j * scale) * xs.key h j d,
   fun h i d => ∑ j, (scoreCotangent (xs.query h) (xs.key h) (xs.value h)
     (xs.outputCotangent h) scale j i * scale) * xs.query h j d,
   fun h i d => ∑ j, probability (xs.query h) (xs.key h) scale j i * xs.outputCotangent h j d⟩

/-- The complete forward/preprocess/backward grids implement causal attention
and its Q/K/V VJP. IO declares all finite FP32 channels, saved scratch and the
initialized in-place DQ alias; source safety and the exact frame are proved. -/
specification kernel_real (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    Spec.Real (PublicIO.logicalRealIO c scale fw pw bw ⊨{PublicIO.logicalZeroDQ c} attentionAndVJP c scale) := by
  exact PublicIO.logical_implements c scale fw pw bw hf hp hb

#stmtSurfaceSubset kernel_real ⊆
  [Shape,Grid.Program,Grid.FullSchedule,Grid.forwardGrid,Grid.preprocessGrid,Grid.backwardGrid,
   Spec.Real,PublicIO.logicalRealIO,LogicalDomainIO.ImplementsOn,PublicIO.logicalZeroDQ,attentionAndVJP,Erased.R,PublicIO.Inputs,PublicIO.Outputs,PublicIO.LogicalInputs,PublicIO.LogicalOutputs,
   MaterializedProgramIO.ImplementsOnCells,PublicIO.realIO]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Correct
