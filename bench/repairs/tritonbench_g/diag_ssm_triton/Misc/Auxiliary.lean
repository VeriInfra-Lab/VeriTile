import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.PerProgram
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.RegionCorrectness
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Mathematics

/-! Retained regional and flat auxiliary results. The public shared IO specifications
are in Correct.lean; these are the exact prior statements with ordinary theorem
status. Complete mathematical correctness of the four selected kernel bodies.
Forward stores h[t+1] for h[0]=s and h[t+1]=h[t]*Lambda+x[t]. Backward
computes gx[t]=gy[t]+carry, adds gx[t]*conj(h[t]) to the per-batch transition
gradient, and sets carry=gx[t]*conj(Lambda). Real paths omit conjugation.
The backward kernel writes per-batch contributions; Torch's later batch sum
is outside kernel scope. Saved y is an explicit backward kernel input.
All contracts include successful execution, each active output and the full
exterior memory frame, including zero time and masked tails. Only this Real
theorem erases dtype. Python assertions establish the public metadata
contract; TensorBridge exposes the actual-coordinate and pointer bindings. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Auxiliary
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

/-- Retained prior regional endpoint: independently bound read-only input views may
overlap physically. Output allocation is supplied at the kernel API boundary. -/
theorem forward_region_correctness (p : ForwardLayout) (length : Nat) :
    Spec.Real (RegionCorrectness.Forward (Kernels.forwardTyped p.s p.x p.l p.y
      length p.batch p.dim p.B).eraseDType p length) :=
  RegionCorrectness.forward p length

theorem backward_region_correctness (p : BackwardLayout) (length : Nat) :
    Spec.Real (RegionCorrectness.Backward (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) :=
  RegionCorrectness.backward p length

theorem complex_forward_region_correctness (p : ForwardLayout) (length : Nat) :
    Spec.Real (RegionCorrectness.ComplexForward (Kernels.complexForwardTyped p.s p.x p.y p.l
      length p.batch p.dim p.B).eraseDType p length) :=
  RegionCorrectness.complex_forward p length

theorem complex_backward_region_correctness (p : BackwardLayout) (length : Nat) :
    Spec.Real (RegionCorrectness.ComplexBackward (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) :=
  RegionCorrectness.complex_backward p length

/-- Additional finite flat result. Unlike the regional endpoint, this legacy
contract requires FlatAlloc.Disjoint, including the input regions. -/
theorem forward_correctness (p : ForwardIO.Layout) (length : Nat) :
    Spec.Real (ForwardIO.Implements (Kernels.forwardTyped p.s p.x p.l p.y
      length p.batch p.dim p.B).eraseDType p length) :=
  PerProgram.forward p length

theorem backward_correctness (p : BackwardIO.Layout) (length : Nat) :
    Spec.Real (BackwardIO.Implements (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) :=
  PerProgram.backward p length

theorem complex_forward_correctness (p : ComplexForwardLoop.Layout) (length : Nat) :
    Spec.Real (ComplexForwardIO.Implements (Kernels.complexForwardTyped p.s p.x p.y p.l
      length p.batch p.dim p.B).eraseDType p length) :=
  PerProgram.complex_forward p length

theorem complex_backward_correctness (p : ComplexBackwardLoop.Layout) (length : Nat) :
    Spec.Real (ComplexBackwardIO.Implements (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) :=
  PerProgram.complex_backward p length

abbrev forward_closed_form := Mathematics.closed_form
abbrev real_batch_adjoint := Mathematics.batch_adjoint
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.Auxiliary
