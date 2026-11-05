import bench.tritonbench_g.layer_norm_triton.Proofs.ForwardEquivalenceIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LayerNormTriton.ForwardPairIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section

/-- The original forward invocation with its complete public IO and observer. -/
def original {α : Type} (m : Model α) (stride N B row : Nat) (eps : ℝ) :=
  ProgramIO.Implementation.original
    (Forward.io m stride N B row eps (fun _ _ => id)) (fun _ => True)

/-- The existing single-iteration loop-unrolling candidate, retaining its
actual invocation and the same complete output observer. -/
def candidate {α : Type} [Inhabited α] (m : Model α)
    (stride N B row : Nat) (eps : ℝ) :=
  ProgramIO.Implementation.ofVariant
    (ForwardEquivalenceIO.candidate m stride N B row eps (fun _ _ => id)) (fun _ => True)

end
end VeriTile.Bench.Repairs.LayerNormTriton.ForwardPairIO
