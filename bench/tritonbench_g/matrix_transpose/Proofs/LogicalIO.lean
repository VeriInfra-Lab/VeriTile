import bench.tritonbench_g.matrix_transpose.IO
import bench.tritonbench_g.matrix_transpose.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.MatrixTranspose.LogicalIOProof
open VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem headline_iff (m d : Nat) :
    (KernelReal.logicalIO m d ⊨ fun xs col row => xs row col) ↔
    (KernelReal.transposeIO m d).Implements (fun _ _ xs j => xs j) := by
  have formula : (KernelReal.logicalIO m d).lift (fun xs col row => xs row col) = id := by
    funext xs i
    exact congrArg xs (Lane2D.encode_decode i)
  change (KernelReal.transposeIO m d).Implements
    (fun _ _ xs => (KernelReal.logicalIO m d).lift (fun xs col row => xs row col) xs) ↔ _
  rw [formula]
  rfl

theorem implements (m d : Nat) :
    (KernelReal.logicalIO m d ⊨ fun xs col row => xs row col) :=
  (headline_iff m d).2 (KernelReal.implements m d)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatrixTranspose.LogicalIOProof
