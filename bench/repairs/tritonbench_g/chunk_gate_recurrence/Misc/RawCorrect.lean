import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReal
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelGrid
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelTail
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Math

/-! Kernel-only correctness. Complete actual typed source execution, every
output and exterior Cell frame share a constructed final state. Optional Tensor
applications and checked bindings are in TensorBridge.lean. Empty calls launch
no kernel. Historical wrapper results remain in Proofs.HostCorrect. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.RawCorrect
noncomputable section
alias chunk_gate_recurrence_kernel_real := KernelReal.complete
alias chunk_gate_recurrence_source_success := KernelExecution.complete
alias chunk_gate_recurrence_adjoint := MathTarget.adjoint_pairing
alias chunk_gate_recurrence_grid_coverage := KernelGrid.coverage
alias chunk_gate_recurrence_tail_untouched := KernelTail.untouched
#axiomsClean chunk_gate_recurrence_kernel_real
#axiomsClean chunk_gate_recurrence_adjoint
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.RawCorrect
