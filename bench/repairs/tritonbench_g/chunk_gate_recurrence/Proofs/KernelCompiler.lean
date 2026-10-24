import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelNative
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCompilerMetadata

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelCompiler
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
noncomputable section
theorem complete (platform : Platform) (binding : Binding platform kind c s seed)
    (p : Program c) (u : State Node) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    ∃ v,Facts platform binding p u v ∧ KernelNative.Outputs
      (platform.instructions binding.object.primitives) (plan platform binding) kind c s seed p u v := by
  obtain ⟨v,facts,outputs⟩ := KernelNative.complete (platform.instructions binding.object.primitives)
    (plan platform binding) kind c s seed p u nonempty hh hk hv
  have same : algebra platform binding=model (platform.instructions binding.object.primitives)
      (plan platform binding) kind c := by
    simp only [algebra,model,binding.operands]
  rw [←same] at facts
  refine ⟨v,⟨facts,?_⟩,outputs⟩
  rw [decoded_source]
  exact facts.success

#axiomsClean complete
#axiomsClean decoded_source
#specNonCircular Binding avoiding [FP.Structural.exec,FP.Structural.run,KernelNative.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelCompiler
