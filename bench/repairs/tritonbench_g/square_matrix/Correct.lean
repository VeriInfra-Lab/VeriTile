import bench.repairs.tritonbench_g.square_matrix.Proofs.KernelReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SquareMatrix.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- One source program squares a contiguous row. The lane mask excludes padding;
input and output windows, source precision erasure and projection are explicit. -/
def squareIO (n B : Nat) : MaskedKernelIO₁ := IO.squareIO n B

/-- Each active output is the square of its independent logical input.
The shared IO judgment includes successful execution, active access safety
and the complete exterior memory frame. -/
specification square_kernel_correctness (n B : Nat) :
    Spec.Real (squareIO n B ⊨ fun xs i => xs i * xs i) :=
  by simpa only [mathematical] using KernelReal.implements n B

#specNonCircular mathematical avoiding [fp32Kernel, indexCommutedKernel]
#axiomsClean square_kernel_correctness
#stmtSurfaceSubset square_kernel_correctness ⊆
  [Spec.Real, squareIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.Correct
