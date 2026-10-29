/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.IOResults

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.FPEquiv
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- Independent native matrix formula: each BK64 block consists of eight
ordered K8 operations with live FP32 seed, then activation and FP16 rounding. -/
def formula (instruction : InstructionProfile) (g : KernelBuffers.Geometry) := fun vals =>
  IO.result g (IO.native (fpReference instruction) g (IO.matrixA g vals) (IO.matrixB g vals))

abbrev kernelIO (M : Algebra Word) (native : PrimitiveOps)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) :=
  IO.matmulIO M native.operations g b

/-- Native observation of the actual complete launched source grid. Existing
primitive ports and nonzero-K scope are unchanged; all raw FP32 operands remain
admitted. The separate zero-fill program stays in the supporting/physical proofs.
No independent numerical rewrite or floating-to-Real identity is asserted. -/
theorem kernel_equiv (M : Algebra Word) (native : PrimitiveOps) (instruction : InstructionProfile)
    (ports : LocalPorts native instruction) (law : Execution.ScalarLaws M native.operations)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (layout : KernelBuffers.Layout g b)
    (positiveK : g.K≠0) :
    (kernelIO M native g b).Implements (formula instruction g) :=
  IOResults.native M native instruction ports law g b layout positiveK

#specNonCircular formula avoiding [Execution.grid,FP.Structural.exec,Source.run]
#axiomsClean kernel_equiv
#stmtSurfaceSubset kernel_equiv ⊆
  [FP.Structural.Algebra,Word,PrimitiveOps,PrimitiveOps.operations,InstructionProfile,LocalPorts,
   Execution.ScalarLaws,KernelBuffers.Geometry,KernelBuffers.Buffers,KernelBuffers.Layout,
   KernelBuffers.Geometry.K,kernelIO,ProgramIO.Implements,formula]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.FPEquiv
