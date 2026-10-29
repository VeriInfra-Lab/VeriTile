import bench.repairs.tritonbench_g.chunk_gate_recurrence.IO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCompilerMetadata

/-! Retain the original selected-object ABI/decoder scope. The invocation uses
the decoded source and decoded operand/reduction choices. No input payload or
output assertion is part of compiler binding metadata. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.NativeIO
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier KernelCompiler
noncomputable section

def compiled (platform : Platform) (binding : Binding platform kind c s seed)
    (p : Program c) : ProgramIO Node Node :=
  { IO.program (model (platform.instructions binding.object.primitives)
      (KernelCompiler.plan platform binding) kind c) kind c s seed p with
    schedule := fun _ => [{ IO.launch (KernelCompiler.algebra platform binding) kind c s seed p with
      kernel := decodedSource platform binding }] }

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.NativeIO
