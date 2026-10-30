import bench.tritonbench_g.matrix_reduction.IO
import bench.tritonbench_g.matrix_reduction.Proofs.IONative
import bench.tritonbench_g.matrix_reduction.Proofs.WrapperFP

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect
open VeriTile.Triton FP.Structural TypedSource
noncomputable section

/-- Successful source-to-native observation with actual access safety and
 the complete typed output/exterior frame. No second kernel or floating rewrite
is asserted: the scalar primitive interpretation is the model boundary. -/
theorem matrix_reduction_kernel_fp {α : Type} [Inhabited α] (A : Algebra α)
    (storage : Storage) (M N : Nat) (hN : 0 < N) :
    (matrixReductionNativeIO A storage M N).Implements (nativeRowMaximum A storage M N) :=
  IONative.correct A storage M N hN

#specNonCircular KernelFP.formula avoiding [kernel, FP.Structural.exec]
#axiomsClean matrix_reduction_kernel_fp
#stmtSurfaceSubset matrix_reduction_kernel_fp ⊆
  [ProgramIO.Implements, matrixReductionNativeIO, nativeRowMaximum,
   FP.Structural.Algebra, Storage]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect
