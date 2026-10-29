import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativeReference
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativeIO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativePublicIO

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.FPEquiv
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
noncomputable section

/-- The retained selected-object result uses the decoded source, ABI and
operand/reduction metadata. Its logical input/output channels are identical. -/
theorem chunk_gate_recurrence_kernel_equiv (platform : KernelCompiler.Platform)
    (binding : KernelCompiler.Binding platform kind c s seed) (p : Program c)
    (nonempty : 0<c.N) :
    (NativeIO.compiled platform binding p).Observes (fun _ _ => Node.value)
      (Reference.native (platform.instructions binding.object.primitives)
        (KernelCompiler.plan platform binding) kind c s seed p) :=
  NativePublicIO.compiled platform binding p nonempty

#axiomsClean chunk_gate_recurrence_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.FPEquiv
