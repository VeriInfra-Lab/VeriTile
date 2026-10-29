import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.APIWords
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.RealContract
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Results

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- All four source-derived formulas agree with the independent public
mathematics at every B,H,T,D coordinate. -/
theorem public_formula (s : Shape) (x : LogicalInputs ℝ s) (kind : OutputKind)
    (i : LogicalIndex s kind) :
    logicalSpec s kind (x.padded realOps) i.bh i.time i.feature=mathematicalResult s x kind i := by
  cases kind with
  | forward =>
    rw [logical_forward_spec]
    simp [mathematicalResult,LogicalInputs.padded,paddedTensor,i.bh_bound,i.head_eq]
  | query =>
    rw [logical_query_spec]
    simp [mathematicalResult,LogicalInputs.padded,paddedTensor,i.bh_bound,i.head_eq]
  | key =>
    rw [logical_key_spec]
    simp [mathematicalResult,LogicalInputs.padded,paddedTensor,i.bh_bound,i.head_eq]
  | value =>
    rw [logical_value_spec]
    simp [mathematicalResult,LogicalInputs.padded,paddedTensor,i.bh_bound,i.head_eq]

theorem forward_api_real (abi : LocalRuntime ℝ σ) (p : Plans) (s : Shape) (domain : ShapeContract s)
    (b : Bases) (tree : SumPlan) (complete : tree.Covers (OutputKind.forward.shards s))
    (layout : ForwardLayout s b) (saved : SavedInputs s b) (state : σ)
    (x : LogicalInputs ℝ s) (bound : ForwardMaterialized s b (abi.observe state) x)
    (i : LogicalIndex s .forward) :
    let result := forwardAPI abi realOps p s b tree layout saved state
    readReturned s .forward result.output (abi.observe result.state) i=mathematicalResult s x .forward i := by
  dsimp only
  rw [forward_api_word]
  have inputs := materialized_forward_reads realOps s b (abi.observe state) x bound
  rw [forward_word_inputs realOps p s tree _ _ inputs.1 inputs.2.1 inputs.2.2]
  exact (wordReference_shards p s .forward tree complete (x.padded realOps) i.bh i.time i.feature
      i.bh_bound i.time.isLt i.feature.isLt).trans
    ((realShards_correct p s .forward (x.padded realOps) i.bh i.time i.feature
      domain.indices.bk_pos domain.indices.bv_pos i.time.isLt i.feature.isLt).trans
      (public_formula s x .forward i))

theorem backward_api_real (abi : LocalRuntime ℝ σ) (p : Plans) (s : Shape) (domain : ShapeContract s)
    (b : Bases) (trees : OutputKind → SumPlan)
    (complete : ∀ kind,kind≠.forward → (trees kind).Covers (kind.shards s))
    (layout : BackwardLayout s b) (context : SavedContext) (state : σ)
    (x : LogicalInputs ℝ s) (bound : Materialized s b (abi.observe state) x)
    (kind : OutputKind) (active : kind≠.forward) (i : LogicalIndex s kind) :
    let result := backwardAPI abi realOps p s b trees layout context state
    readReturned s kind (result.handle kind) (abi.observe result.state) i=mathematicalResult s x kind i := by
  dsimp only
  rw [backward_api_word abi realOps p s b trees layout context state kind active,
    materialized_read realOps s b (abi.observe state) x bound,
    wordReference_shards p s kind (trees kind) (complete kind active) (x.padded realOps) i.bh i.time i.feature
      i.bh_bound i.time.isLt i.feature.isLt,
    realShards_correct p s kind (x.padded realOps) i.bh i.time i.feature
      domain.indices.bk_pos domain.indices.bv_pos i.time.isLt i.feature.isLt]
  exact public_formula s x kind i


#axiomsClean public_formula
#axiomsClean forward_api_real
#axiomsClean backward_api_real

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
