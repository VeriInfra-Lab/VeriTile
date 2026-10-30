import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.matrix_transpose.Kernels
import bench.tritonbench_g.matrix_transpose.Proofs.Mathematical_Source
import bench.tritonbench_g.matrix_transpose.Proofs.Mathematical_Real

namespace VeriTile.Bench.Repairs.MatrixTranspose.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatrixTranspose
noncomputable section

/-- The canonical typed source, projected only for mathematical semantics.
The transpose is explicit in the source and output windows. -/
def transposeIO (m d : Nat) : Masked2DKernelIO₁ where
  kernel := (source "matrix" "out" m d).eraseDType
  projection := typed_supported Bool.false "matrix" "out" (max d 1) 1 (max m 1) 1 m d
  inp := "matrix"
  out := "out"
  B := d*m
  read := fun _ _ j => matrixAddr (max d 1) 1 (Lane2D.decode j)
  write := fun _ _ j => outAddr (max m 1) 1 (Lane2D.decode j)
  mask := fun _ _ _ => True

end
end VeriTile.Bench.Repairs.MatrixTranspose.KernelReal

namespace VeriTile.Bench.Repairs.MatrixTranspose.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatrixTranspose
noncomputable section

/-- The input window's lane order viewed as logical matrix rows and columns. -/
def matrixInputs (m d : Nat) :
    (Fin (d*m) → ℝ) ≃ (Fin m → Fin d → ℝ) where
  toFun := fun xs row col => xs (Lane2D.encode (col,row,PUnit.unit))
  invFun := fun xs i => xs (Lane2D.decode i).2.1 (Lane2D.decode i).1
  left_inv := by
    intro xs
    funext i
    exact congrArg xs (Lane2D.encode_decode i)
  right_inv := by
    intro xs
    funext row col
    simp only [Lane2D.decode_encode]

/-- The output window's same lanes viewed as transposed matrix coordinates. -/
def matrixOutputs (m d : Nat) :
    (Fin (d*m) → ℝ) ≃ (Fin d → Fin m → ℝ) where
  toFun := fun ys col row => ys (Lane2D.encode (col,row,PUnit.unit))
  invFun := fun ys i => ys (Lane2D.decode i).1 (Lane2D.decode i).2.1
  left_inv := by
    intro ys
    funext i
    exact congrArg ys (Lane2D.encode_decode i)
  right_inv := by
    intro ys
    funext col row
    simp only [Lane2D.decode_encode]

end
end VeriTile.Bench.Repairs.MatrixTranspose.KernelReal

namespace VeriTile.Bench.Repairs.MatrixTranspose.KernelReal
open VeriTile.Triton
noncomputable section

/-- Same complete two-PID judgment, with logical input/output matrix coordinates. -/
def logicalIO (m d : Nat) :
    LogicalIO (Fin (d*m) → ℝ) (Fin (d*m) → ℝ)
      (Fin m → Fin d → ℝ) (Fin d → Fin m → ℝ)
      (fun f => (transposeIO m d).Implements (fun _ _ xs => f xs)) where
  inputs := matrixInputs m d
  outputs := matrixOutputs m d

end
end VeriTile.Bench.Repairs.MatrixTranspose.KernelReal
