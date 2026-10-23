import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Correct
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.KernelContract

/-! Optional Python-check correspondence and Tensor argument applications.
The primary mathematical and FP entries do not import this module. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.TensorBridge
open VeriTile.Triton VeriTile.Torch
noncomputable section

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype=.ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

/-- The input assertions and ordinary Tensor binding construct the native PID;
all remaining signed-i64 control, address and byte-span facts are proved. -/
theorem native_programs (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) (nonempty : 0<h.B*h.H*h.T)
    (c : Config) (chosen : c∈configurations) (row : Fin (h.B*h.H)) :
    NativeIndex.ProgramCertificate h c p.backing (binding.pid s dtype contract h p nonempty row) :=
  NativeIndex.certify h c chosen p.backing _

#axiomsClean python_checks_iff_preconditions
#axiomsClean native_programs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.TensorBridge
