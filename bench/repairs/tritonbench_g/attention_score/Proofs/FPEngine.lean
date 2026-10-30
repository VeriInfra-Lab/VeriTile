import bench.repairs.tritonbench_g.attention_score.Proofs.RealSource
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.AttentionScore.FPEngine
open VeriTile.Triton FP.Structural
noncomputable section

/-- Internal total operation interpretation used to prove the actual typed
program runs for arbitrary symbolic values. Public native results instantiate
this with an explicit fixed-instruction graph, never a numerical whole-dot
equality assumption. No numerical laws are attached to this structure. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) →
    Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def withDot {α : Type} (M : Algebra α) (D : Dot α) : Algebra α :=
  {M with
    dot := fun p {_} {_} {_} {_} a b => some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p options {_} {_} {_} {_} a b => some (D.apply p .real .real .real options a b)
    dotTyped := fun p l r o options {_} {_} {_} {_} a b => some (D.apply p l r o options a b)}

def kernel (B D : Nat) (p : RealSource.Parameters) : ComputeKernel :=
  RealSource.source B D p Bool.false

def parts (B D : Nat) (p : RealSource.Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel B D p).surfaceBody "start_m").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.FPEngine
