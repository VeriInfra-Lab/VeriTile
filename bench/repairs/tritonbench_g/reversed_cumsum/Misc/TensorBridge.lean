import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.KernelCompatibility
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPCompatibility

/-! Optional Tensor/check applications; the primary kernel specifications
remain in Correct.lean and FPEquiv.lean, which do not import this bridge. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.TensorBridge
open VeriTile.Triton VeriTile.Torch
noncomputable section

/-- The original physical kernel result, including stable input allocation
and the full exterior element-cell frame. -/
alias kernel_correct := KernelCompatibility.physical_kernel_correct

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype=.ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

theorem native_programs (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) (nonempty : 0<h.B*h.H*(h.T*h.S))
    (c : Config) (chosen : c∈configurations) (program : Geometry.Program h) :
    NativeIndex.ProgramCertificate h c p.backing (binding.pid s dtype contract h p nonempty program) :=
  NativeIndex.certify h c chosen p.backing _

#axiomsClean kernel_correct
#axiomsClean python_checks_iff_preconditions
#axiomsClean native_programs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.TensorBridge
