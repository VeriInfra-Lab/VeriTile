import bench.tritonbench_g.parallel_attention.ScalarModel

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open scoped BigOperators
noncomputable section

def realScalars : Scalars ℝ where
  literal := id
  add := (·+·)
  mul := (·*·)
  fma := fun x y z => x*y+z
  cast := id
  div := (·/·)
  neg := Neg.neg

/-- The cotangent attached to one value shard. DZ belongs only to shard zero. -/
def shardCotangent (shape : Shape) (x : TensorInputs ℝ) (bh shard t j : Nat) : ℝ :=
  (∑ p∈Finset.range shape.BV,
    if shard*shape.BV+p<shape.V then x.do_ bh t (shard*shape.BV+p)*x.v bh j (shard*shape.BV+p) else 0)+
    (if shard=0 then x.dz bh t else 0)

/-- Independent derivative of squared causal attention for the DQ scratch slice. -/
def queryPartial (shape : Shape) (σ : ℝ) (x : TensorInputs ℝ) (bh shard t e : Nat) : ℝ :=
  σ*∑ j∈Finset.range shape.T,
    if j≤t then 2*shardCotangent shape x bh shard t j*
      score shape.K σ (headInputs x bh) t j*x.k bh j e else 0

/-- Independent derivative for the DK scratch slice; no host shard sum. -/
def keyPartial (shape : Shape) (σ : ℝ) (x : TensorInputs ℝ) (bh shard j e : Nat) : ℝ :=
  ∑ t∈Finset.range shape.T,
    if t<shape.T ∧ j≤t then 2*σ*shardCotangent shape x bh shard t j*
      score shape.K σ (headInputs x bh) t j*x.q bh t e else 0

/-- Complete DV, since the checked K domain has exactly one key shard. -/
def valueGradient (shape : Shape) (σ : ℝ) (x : TensorInputs ℝ) (bh j p : Nat) : ℝ :=
  ∑ t∈Finset.range shape.T,
    if t<shape.T ∧ j≤t then (score shape.K σ (headInputs x bh) t j)^2*x.do_ bh t p else 0

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
