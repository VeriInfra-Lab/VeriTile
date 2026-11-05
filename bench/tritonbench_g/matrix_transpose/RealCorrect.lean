import bench.tritonbench_g.matrix_transpose.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MatrixTranspose.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Complete original single tile with logical input `(row,col)` and output
`(col,row)` coordinates. Both PID axes, access safety and frame are unchanged. -/
def transposeIO (m d : Nat) := KernelReal.logicalIO m d
specification transpose_kernel_correctness (m d : Nat) (_h : InputContract m d) :
    Spec.Real (transposeIO m d ⊨ fun xs col row => xs row col) :=
  LogicalIOProof.implements m d

#specNonCircular formula avoiding [source,indexCommutedSource]
#stmtSurfaceSubset transpose_kernel_correctness ⊆
  [Spec.Real,transposeIO,LogicalIO.Implements,InputContract,Masked2DKernelIO₁.Implements,KernelReal.transposeIO,Masked2DKernelIO₁.B]
#axiomsClean transpose_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.Correct
