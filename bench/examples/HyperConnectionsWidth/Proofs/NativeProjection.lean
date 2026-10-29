import bench.examples.HyperConnectionsWidth.Kernels
import bench.examples.HyperConnectionsWidth.Proofs.MathKernels
import VeriTile.Meta.StatementAudit

/- Complete native mathematical projections. No prefix or selected output
replaces a full program; every iteration and memory statement is retained. -/
namespace VeriTile.Bench.Examples.HyperConnectionsWidth.NativeProjection
open VeriTile Triton

set_option maxHeartbeats 6400000
set_option linter.unusedSimpArgs false

@[simp] theorem matrixKernel_projection (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.matrixKernel resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel =
    (MathKernels.matrixKernel resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel := by
  rfl
#axiomsClean matrixKernel_projection

@[simp] theorem scalarKernel_projection (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (tau : ℝ) :
  (Kernels.scalarKernel resReg hResReg hPreReg resMixReg branchInReg tau).toAlgKernel =
    (MathKernels.mhcWidthConnectionKernel resReg hResReg hPreReg resMixReg branchInReg 1 1 1 0 tau).toAlgKernel := by
  rfl
#axiomsClean scalarKernel_projection

@[simp] theorem mhcWidthConnectionKernel_projection (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.mhcWidthConnectionKernel resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel =
    (MathKernels.mhcWidthConnectionKernel resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel := by
  unfold Kernels.mhcWidthConnectionKernel
  split <;> simp_all only [scalarKernel_projection, matrixKernel_projection,
    MathKernels.mhcWidthConnectionKernel]
#axiomsClean mhcWidthConnectionKernel_projection

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

@[simp] theorem matrixOptimized_projection (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.matrixOptimized resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel =
    (MathKernels.matrixOptimized resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel := by
  rfl
#axiomsClean matrixOptimized_projection

@[simp] theorem middleKernel_projection (tau : ℝ) :
  (Kernels.middleKernel tau).toAlgKernel =
    (MathKernels.middleKernel tau).toAlgKernel := by
  rfl
#axiomsClean middleKernel_projection

@[simp] theorem matrixMiddle_projection (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) :
  (Kernels.matrixMiddle resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel =
    (MathKernels.matrixMiddle resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau).toAlgKernel := by
  rfl
#axiomsClean matrixMiddle_projection
end VeriTile.Bench.Examples.HyperConnectionsWidth.NativeProjection
