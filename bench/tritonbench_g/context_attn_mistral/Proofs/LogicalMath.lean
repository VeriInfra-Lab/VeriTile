import bench.tritonbench_g.context_attn_mistral.IO

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.TensorMath
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open scoped BigOperators
noncomputable section

/-- Exact rounded source key-block count, including original underlaunch. -/
def keyCount (c : Host.Call) (capacity : RegionBounds) (xs : LogicalData.Inputs c capacity)
    (j : LogicalData.Coordinate c) : Nat :=
  if (j.block.val*c.layout.BM : Int) < LogicalData.sequenceLength xs j.batch.val then
    ((j.block.val+1)*c.layout.BM) / c.layout.BN +
      if ((j.block.val+1)*c.layout.BM) % c.layout.BN = 0 then 0 else 1
  else 0

/-- Independent causal/window dot-product score over batch/head/time/dimension. -/
def attentionScore (c : Host.Call) (capacity : RegionBounds) (xs : LogicalData.Inputs c capacity)
    (j : LogicalData.Coordinate c) (key : Nat) : ℝ :=
  let queryTime := j.block.val*c.layout.BM+j.row.val
  if (key : Int) ≤ queryTime ∧ (queryTime : Int)-c.layout.window < key then
    (∑ d : Fin c.layout.D,
      xs.query j.batch.val j.head.val queryTime d.val *
        xs.key j.batch.val (j.head.val/c.layout.kvGroup) key d.val) * Host.scale c.layout.D
  else -1000000000

/-- Gathered finite-sentinel weighted sum on the complete original grid/lane
window. Logical tensor views preserve masked/padded values and finite scores. -/
def finiteSentinelAttention (c : Host.Call) (capacity : RegionBounds)
    (xs : LogicalData.Inputs c capacity) : LogicalData.Coordinate c → ℝ :=
  fun j =>
    (∑ key : Fin (keyCount c capacity xs j * c.layout.BN),
      Real.exp (attentionScore c capacity xs j key.val) *
        xs.value j.batch.val (j.head.val/c.layout.kvGroup) key.val j.dim.val) /
    ∑ key : Fin (keyCount c capacity xs j * c.layout.BN),
      Real.exp (attentionScore c capacity xs j key.val)

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.TensorMath
