import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelReal

/-! Kernel-only correctness of the three exact FP32 source templates under
their mathematical interpretation. The successful source execution, every
output and exterior Cell frame share the constructed final state. Optional
Tensor applications and pointer/check correspondences are in TensorBridge.
Historical wrapper allocation/copyback proofs remain in Proofs.HostCorrect. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelRealResults
noncomputable section
abbrev decay_cumsum_kernel_real := @KernelReal.complete
abbrev decay_cumsum_source_success := @KernelExecution.phase_complete
#axiomsClean decay_cumsum_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelRealResults
