import bench.repairs.tritonbench_g.matmul_triton1.IO
import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOCorrect
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.Native

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.MatmulTriton1
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- Complete source-to-native FP32 result under the original explicit scalar
instruction model: live carry, ascending FMA lanes and every masked padding
lane. This is a native primitive boundary, with no optimization rewrite or
FP-to-Real equality. The existing local IEEE transport is in Proofs/IONative. -/
theorem matmul_triton1_native (rest : Algebra Native.Word) (actual : Native.Instructions)
    (p : Core.Parameters) (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (MatmulIO.io rest (Native.primitives actual) p).Implements
      (MatmulIO.precisionTarget (Native.primitives actual) p) :=
  IOCorrect.precision rest (Native.primitives actual) p canonical ready storage geometry ha hb

#axiomsClean matmul_triton1_native
#specNonCircular MatmulIO.precisionTarget avoiding [Core.source,Kernel.source,Core.grid,FiniteSource.grid,FP.Structural.exec]
#stmtSurfaceSubset matmul_triton1_native ⊆
  [ProgramIO.Implements,MatmulIO.io,MatmulIO.precisionTarget,
   FP.Structural.Algebra,Native.Word,Native.Instructions,Native.primitives,instInhabitedWord,
   Core.Parameters,Core.Parameters.input,Core.Parameters.weight,Core.Parameters.output,
   FiniteSource.Canonical,FiniteIndex.Ready,FiniteIndex.StorageBounds,Geometry.Valid,Core.geometry,RegionName]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulTriton1
