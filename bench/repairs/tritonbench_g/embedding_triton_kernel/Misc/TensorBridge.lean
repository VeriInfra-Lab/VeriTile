import bench.repairs.tritonbench_g.embedding_triton_kernel.Correct

/-! Optional Python admission and Tensor metadata-to-kernel contract bridge. -/
namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff_preconditions (a : Arguments) :
    checkArguments a = Bool.true ↔ AdmittedArguments a := argument_checks_iff a

export Bounds (kernel_contract)

#axiomsClean python_checks_iff_preconditions
#axiomsClean kernel_contract
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.TensorBridge
