import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardIO

/-! Reuse of the four unchanged complete per-program Real execution proofs.
Integer casts are discharged by the separate native-index bridge. These are
kernel-only theorems. Historical host composition remains optional; it is
not imported or required by the public kernel correctness entry. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.PerProgram
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section

theorem forward (p : ForwardIO.Layout) (length : Nat) :
    Spec.Real (ForwardIO.Implements (Kernels.forwardTyped p.s p.x p.l p.y
      length p.batch p.dim p.B).eraseDType p length) :=
  ForwardIO.original_correct .fp32 p length

theorem backward (p : BackwardIO.Layout) (length : Nat) :
    Spec.Real (BackwardIO.Implements (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) :=
  BackwardIO.original_correct .fp32 p length

theorem complex_forward (p : ComplexForwardLoop.Layout) (length : Nat) :
    Spec.Real (ComplexForwardIO.Implements (Kernels.complexForwardTyped p.s p.x p.y p.l
      length p.batch p.dim p.B).eraseDType p length) :=
  ComplexForwardIO.original_correct p length

theorem complex_backward (p : ComplexBackwardLoop.Layout) (length : Nat) :
    Spec.Real (ComplexBackwardIO.Implements (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) :=
  ComplexBackwardIO.original_correct p length

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.PerProgram
