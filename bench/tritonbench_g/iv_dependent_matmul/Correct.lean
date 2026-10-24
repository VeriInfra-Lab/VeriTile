import bench.tritonbench_g.iv_dependent_matmul.Kernels
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Real
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Safety
import bench.tritonbench_g.iv_dependent_matmul.Proofs.IndexOverflow
import bench.tritonbench_g.iv_dependent_matmul.Proofs.LegacyCorrect

/-! Independent Real target `sum_{k<K} A[i,k]*B[k,j]`, complete ceil-K tail
contraction, and whole-host composition. `NumericalBoundary` explicitly lists
finite reached FP32 accumulators, exact dot/add projection on their actual
operands, and final half narrowing. These numerical obligations are deferred;
no execution, final-memory or returned-output premise is used.

The original source has an independently checked signed-index defect on a
valid large FP32 output allocation. Bounded-domain proofs and all historical
specifications remain valid, but whole-task acceptance is zero pending repair.
See Proofs/IndexOverflow.lean and the unchanged-source compiler/replay receipts. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Correct
open VeriTile.Triton FP.Structural FPValue Kernels Whole RealTarget
noncomputable section

specification independent_matrix_product_tail (inner width : Nat) (hw : 0<width)
    (A B : Nat → Nat → ℝ) (i j : Nat) :
    (∑ k∈Finset.range ((inner+width-1)/width),blockSum inner width A B i j k)=
      mathematical inner A B i j := masked_blocks_matmul inner width hw A B i j

specification fp32_whole_host_real_projection [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (graph : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (indices : IndexDomain r s allocation) (roundHalf : ℝ → ℝ)
    (numerical : NumericalBoundary cfg D F graph r s roundHalf) :
    ∃ final, host rest cfg D F graph r s=some (outputMetadata r,final) ∧
      (∀ i<r.rows, ∀ j<r.columns, realValue ((final.mem r.C (offset r i j)).read .fp32)=
        roundHalf (mathematical r.inner (inputA r s) (inputB r s) i j)) ∧
      (∀ region o, ¬Footprint r region o → final.mem region o=s.mem region o) ∧
      SourceSafety r s allocation := by
  obtain ⟨final,he,hv,hf⟩ := host_projection rest cfg D F graph r s allocation roundHalf numerical
  exact ⟨final,he,hv,hf,source_safety r s allocation indices⟩

#axiomsClean independent_matrix_product_tail
#axiomsClean fp32_whole_host_real_projection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Correct
