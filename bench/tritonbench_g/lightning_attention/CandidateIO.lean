import bench.tritonbench_g.lightning_attention.Proofs.PairSource
import bench.tritonbench_g.lightning_attention.NativeIO

/-! Candidate invocation and materialized schedule on the unchanged native IO. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairExecution
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (D : DotPrimitive α) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) : Invocation α :=
  { KernelOnly.invocation M D stage d pid with kernel := PairSource.kernel stage d.parameters }

/-- Exactly the original codec and per-CTA materialization boundary. -/
def launch (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) : MaterializedInvocation Word Native.Expr :=
  { NativeIO.launch ops plan stage d pid with
    source := invocation Native.algebra Native.dots stage d pid }

def schedule (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) := (grid stage d).map (launch ops plan stage d)

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairExecution
