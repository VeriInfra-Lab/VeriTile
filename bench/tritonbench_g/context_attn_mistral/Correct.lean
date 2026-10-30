import bench.tritonbench_g.context_attn_mistral.RealIO
import bench.tritonbench_g.context_attn_mistral.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open scoped BigOperators
open scoped VeriTile.Triton.LogicalDomainIO
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

/-- Complete ordinary grid implements the independent finite-sentinel weighted
sum. Actual source access safety, successful execution and exterior memory frame
are conclusions of the shared IO judgment. Original underlaunch is retained. -/
specification kernel_real (c : Host.Call) (capacity : RegionBounds)
    (geometry : RealInterface.Geometry c capacity) :
    Spec.Real ((LogicalData.attentionIO c capacity) ⊨{LogicalData.Metadata c capacity} (finiteSentinelAttention c capacity)) :=
  LogicalData.logical_correct c capacity geometry

#specNonCircular finiteSentinelAttention avoiding [exec,execR,AlgorithmExecution]
#axiomsClean kernel_real

#guard_msgs (drop info) in
#auditModuleAxioms
#stmtSurfaceSubset kernel_real ⊆
  [VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData.Coordinate,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData.Inputs,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData.Metadata,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData.attentionIO,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface.Geometry,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface.Inputs,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface.laneCount,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface.programIO,
   VeriTile.Bench.Repairs.ContextAttnMistralKernel.finiteSentinelAttention,
   VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host.Call,
   VeriTile.Spec.Real,
   VeriTile.Triton.AlgorithmProgramIO.ImplementsOn,
   VeriTile.Triton.LogicalDomainIO.ImplementsOn,
   VeriTile.Triton.RegionBounds]

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel
