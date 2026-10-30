import bench.repairs.tritonbench_g.parallel_retention_attention.IO
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.FusedLocal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
open VeriTile Triton FP.Structural Kernel Kernel.Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (withDot)
noncomputable section
variable {α : Type}
def forwardInvocation (s : Shape) (tile : Tile) : Invocation (Expr α) :=
  { PublicIO.forwardInvocation α s tile with kernel:=forward (config s 128) }

def backwardInvocation (s : Shape) (tile : Tile) : Invocation (Expr α) :=
  { PublicIO.backwardInvocation α s tile with
    kernel:=FPBackward.fusedShellSource (decodedBackwardConfig s (PublicIO.programIds s tile)) "dq" }

def forwardVariant (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO.Variant (PublicIO.forwardIO a p s tile) where
  schedule:=fun _=>[forwardInvocation s tile]
  observe:=fun _ _=>evaluate a p

def backwardVariant (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO.Variant (PublicIO.backwardIO a p s tile) where
  schedule:=fun _=>[backwardInvocation s tile]
  observe:=fun _ _=>evaluate a p

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
