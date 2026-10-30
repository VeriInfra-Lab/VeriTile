import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open VeriTile.Bench.Repairs.ParallelAttention IOData
open scoped BigOperators
noncomputable section
/-- Logical score, causal square weight, and value-shard cotangent. DZ is
attached only to shard zero; DQ/DK remain partial gradients for that shard. -/
def squaredScore (shape : Shape) (scale : ℝ) (x : TensorInputs ℝ) (bh t j : Nat) : ℝ :=
  scale * ∑ e∈Finset.range shape.K, x.q bh t e*x.k bh j e

def causalWeight (shape : Shape) (scale : ℝ) (x : TensorInputs ℝ) (bh t j : Nat) : ℝ :=
  if j≤t then (squaredScore shape scale x bh t j)^2 else 0

def cotangent (shape : Shape) (x : TensorInputs ℝ) (bh shard t j : Nat) : ℝ :=
  (∑ p∈Finset.range shape.BV,
    if shard*shape.BV+p<shape.V then x.do_ bh t (shard*shape.BV+p)*x.v bh j (shard*shape.BV+p) else 0)+
    (if shard=0 then x.dz bh t else 0)

/-- Logical outputs retain every actual value shard, including partial DQ/DK. -/
def mathematical (s : Shape) (scale : ℝ) (xs : LogicalView.Inputs s) : LogicalView.Outputs s :=
  let x := LogicalView.tensors s xs
  { value := fun bh shard row column => ∑ n∈Finset.range s.T,
      causalWeight s scale x bh.val row.val n*x.v bh.val n (shard.val*s.BV+column.val)
    normalization := fun bh _ row => ∑ n∈Finset.range s.T,
      causalWeight s scale x bh.val row.val n
    queryGradient := fun bh shard row column => scale*∑ n∈Finset.range s.T,
      if n≤row.val then 2*cotangent s x bh.val shard.val row.val n*
        squaredScore s scale x bh.val row.val n*x.k bh.val n column.val else 0
    keyGradient := fun bh shard row column => ∑ t∈Finset.range s.T,
      if t<s.T ∧ row.val≤t then 2*scale*cotangent s x bh.val shard.val t row.val*
        squaredScore s scale x bh.val t row.val*x.q bh.val t column.val else 0
    valueGradient := fun bh shard row column => ∑ t∈Finset.range s.T,
      if t<s.T ∧ row.val≤t then (squaredScore s scale x bh.val t row.val)^2*
        x.do_ bh.val t (shard.val*s.BV+column.val) else 0 }

specification forward_grid_real (shape : Shape) (accepted : ShapeContract shape) (scale : ℝ) :
    Spec.Real (LogicalView.attentionIO .forward shape scale ⊨ mathematical shape scale) :=
  (LogicalReader.judgment_iff .forward shape scale).mpr (ReaderBefore.forward_grid_real shape accepted scale)

specification backward_grid_real (shape : Shape) (accepted : ShapeContract shape) (scale : ℝ) :
    Spec.Real (LogicalView.attentionIO .backward shape scale ⊨ mathematical shape scale) :=
  (LogicalReader.judgment_iff .backward shape scale).mpr (ReaderBefore.backward_grid_real shape accepted scale)

#axiomsClean forward_grid_real
#axiomsClean backward_grid_real
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Correct
