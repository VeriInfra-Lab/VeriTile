import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Contract
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Specification

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- The independent mathematical public result. On the FP32 implementation,
its relation to native words is a numerical obligation, not an equality of
floating-point arithmetic and real arithmetic. -/
def mathematicalResult (s : Shape) (x : LogicalInputs ℝ s) (kind : OutputKind)
    (i : LogicalIndex s kind) : ℝ :=
  let bh : Fin (s.B*s.H) := ⟨i.bh,i.bh_bound⟩
  match kind with
  | .forward => outputSpec s.T s.DK s.DV i.head (x.q bh) (x.k bh) (x.v bh) i.time i.feature
  | .query => querySpec s.T s.DK s.DV i.head (x.k bh) (x.v bh) (x.g bh) i.time i.feature
  | .key => keySpec s.T s.DK s.DV i.head (x.q bh) (x.v bh) (x.g bh) i.time i.feature
  | .value => valueSpec s.T s.DK s.DV i.head (x.q bh) (x.k bh) (x.g bh) i.time i.feature

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
