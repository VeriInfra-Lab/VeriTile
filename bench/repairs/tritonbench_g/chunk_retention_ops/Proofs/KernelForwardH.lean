import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardO
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHComplete
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelForward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def kernel (c : Shape) (seed final : Bool) :=
  Source.forwardH "k" "v" "h" "h0" "ht"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    c.H c.T c.K c.V 64 c.BK c.BV c.chunks seed final

theorem source_identity (c : Shape) (seed final : Bool) : kernel c seed final=
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource.kernel (KernelOutput.shape c) seed final := rfl

variable {α : Type} [Inhabited α]
theorem program (M : Algebra α) (D : KernelOutput.Shared.DotPrimitive α) (c : Shape)
    (seed final : Bool) (s : State α) :
    ∃ t,FP.Structural.exec (engine (KernelOutput.Shared.withDot M D)) (kernel c seed final) s=some t ∧
      t.mem=(VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource.written M D
        (KernelOutput.shape c) seed final s).mem := by
  rw [source_identity]
  exact VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource.source_memory M D
    (KernelOutput.shape c) seed final s

#axiomsClean source_identity
#axiomsClean program
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelForward
