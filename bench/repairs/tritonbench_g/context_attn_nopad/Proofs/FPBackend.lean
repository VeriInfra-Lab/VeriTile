import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Source
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPBackend
open VeriTile.Triton FP.Structural
noncomputable section

/-- Only the two source default-dot sites need optional backend support in
the selected FP32 AST. The callback is arbitrary: availability imposes no
numerical equation, successful-kernel assumption or output law. All scalar,
cast, max and sum callbacks remain the original opaque Algebra operations. -/
structure Ops (α : Type) where
  scalar : Algebra α
  dot : Option ComputeDType → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m,k]) → Values α .real (batch ++ [k,n]) →
    Values α .real (batch ++ [m,n])

def Ops.algebra {α : Type} (o : Ops α) : Algebra α :=
  { o.scalar with dot := fun p {_} {_} {_} {_} a b => some (o.dot p a b) }

end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPBackend
