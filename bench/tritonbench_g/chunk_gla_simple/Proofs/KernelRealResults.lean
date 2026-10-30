import bench.tritonbench_g.chunk_gla_simple.Proofs.Static

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.KernelRealResults
open VeriTile Triton
open Host
noncomputable section

/-- Successful readable source grid, independent initial-input Real formula,
and the complete tagged-memory frame. There is no source-success premise. -/
theorem whole_kernel (c : Shape) (bt : 0<c.BT) (scale : ℝ) (before : BlockState) :
    Nonempty (Kernel.GridLaunchedOrdinary (DirectReal.kernel c scale) (grid c) before
      (DirectReal.result c scale before)) ∧
    (∀ h t v,(DirectReal.result c scale before).readMem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)=
      Host.formula c bt (VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real.inputs regions before) scale h t v) ∧
    BlockState.WriteWithin (DirectReal.footprint c) before (DirectReal.result c scale before) :=
  DirectReal.complete c bt scale before

#axiomsClean whole_kernel
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.KernelRealResults
