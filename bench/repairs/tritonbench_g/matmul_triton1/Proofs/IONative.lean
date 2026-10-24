import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOCorrect
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.Native

namespace VeriTile.Bench.Repairs.MatmulTriton1.IONative
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- The existing conditional scalar zero/FMA transport, now also available on
the complete shared IO judgment. It adds no numeric law or finite-value guard. -/
theorem fp32_if_local (rest : Algebra Native.Word) (actual : Native.Instructions)
    (numeric : Native.Correspondence actual) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (MatmulIO.io rest (Native.primitives actual) p).Implements
      (MatmulIO.precisionTarget (Native.primitives Native.referenceInstructions) p) := by
  rw [Native.instruction_eq actual numeric]
  exact IOCorrect.precision rest (Native.primitives Native.referenceInstructions) p
    canonical ready storage geometry ha hb

#axiomsClean fp32_if_local
end
end VeriTile.Bench.Repairs.MatmulTriton1.IONative
