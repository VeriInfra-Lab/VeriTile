import bench.repairs.tritonbench_g.attention_kernel.Proofs.PairPrefix
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.AttentionKernel.Pair
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section

def job {β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters) (pid : Program) : Invocation β :=
  {IO.job M dot p pid with kernel := kernel p}

def variant {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters)
    (work : List Program) (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α) :
    ProgramIO.Variant (IO.attentionIO M dot p work encode observe) where
  schedule := fun _ => work.map (job M dot p)
  observe := (IO.attentionIO M dot p work encode observe).observe

def original {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters)
    (work : List Program) (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α) :=
  ProgramIO.Implementation.original (IO.attentionIO M dot p work encode observe) (fun _ => True)

def queryEarly {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters)
    (work : List Program) (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α) :=
  ProgramIO.Implementation.ofVariant (variant M dot p work encode observe) (fun _ => True)

end
end VeriTile.Bench.Repairs.AttentionKernel.Pair
