import bench.examples.HyperConnectionsDepth.Kernels
import bench.examples.HyperConnectionsDepth.Proofs.MathKernels
import VeriTile.Meta.StatementAudit

/- Complete native mathematical projections. No prefix or selected output
replaces a full program; every iteration and memory statement is retained. -/
namespace VeriTile.Bench.Examples.HyperConnectionsDepth.NativeProjection
open VeriTile Triton

set_option maxHeartbeats 6400000
set_option linter.unusedSimpArgs false

@[simp] theorem matrixKernel_projection (resMixReg branchOutReg hPostReg outReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.matrixKernel resMixReg branchOutReg hPostReg outReg S T D numIters tau).toAlgKernel =
    (MathKernels.matrixKernel resMixReg branchOutReg hPostReg outReg S T D numIters tau).toAlgKernel := by
  rfl
#axiomsClean matrixKernel_projection

@[simp] theorem scalarKernel_projection (resMixReg branchOutReg hPostReg outReg : RegionName) (tau : ℝ) :
  (Kernels.scalarKernel resMixReg branchOutReg hPostReg outReg tau).toAlgKernel =
    (MathKernels.mhcDepthConnectionKernel resMixReg branchOutReg hPostReg outReg 1 1 1 0 tau).toAlgKernel := by
  rfl
#axiomsClean scalarKernel_projection

@[simp] theorem mhcDepthConnectionKernel_projection (resMixReg branchOutReg hPostReg outReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.mhcDepthConnectionKernel resMixReg branchOutReg hPostReg outReg S T D numIters tau).toAlgKernel =
    (MathKernels.mhcDepthConnectionKernel resMixReg branchOutReg hPostReg outReg S T D numIters tau).toAlgKernel := by
  unfold Kernels.mhcDepthConnectionKernel
  split <;> simp_all only [scalarKernel_projection, matrixKernel_projection,
    MathKernels.mhcDepthConnectionKernel]
#axiomsClean mhcDepthConnectionKernel_projection

@[simp] theorem originalKernel_projection (tau : ℝ) :
  (Kernels.originalKernel tau).toAlgKernel =
    (MathKernels.originalKernel tau).toAlgKernel := by
  rfl
#axiomsClean originalKernel_projection

@[simp] theorem optimizedKernel_projection (tau : ℝ) :
  (Kernels.optimizedKernel tau).toAlgKernel =
    (MathKernels.optimizedKernel tau).toAlgKernel := by
  rfl
#axiomsClean optimizedKernel_projection

@[simp] theorem matrixOriginal_projection (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.matrixOriginal S T D numIters tau).toAlgKernel =
    (MathKernels.matrixOriginal S T D numIters tau).toAlgKernel := by
  rfl
#axiomsClean matrixOriginal_projection

@[simp] theorem matrixOptimized_projection (resMixReg branchOutReg hPostReg outReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.matrixOptimized resMixReg branchOutReg hPostReg outReg S T D numIters tau).toAlgKernel =
    (MathKernels.matrixOptimized resMixReg branchOutReg hPostReg outReg S T D numIters tau).toAlgKernel := by
  rfl
#axiomsClean matrixOptimized_projection
end VeriTile.Bench.Examples.HyperConnectionsDepth.NativeProjection
