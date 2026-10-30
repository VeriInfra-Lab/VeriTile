import bench.repairs.tritonbench_g.context_attn_mistral.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ContextAttnMistral.FPBackend
open VeriTile.Triton FP.Structural
noncomputable section

/-- Available local primitives on an arbitrary floating carrier. These are
functions, not numerical equations or whole-program execution assumptions.
Distinct dot, typed-dot, and typed reductions retain their declared options,
shapes and channels. No FMA/dotAcc operation is substituted for source add. -/
structure Ops (α : Type) where
  scalar : Algebra α
  eq : Option ComputeDType → FloatDType → α → α → Bool
  dot : Option ComputeDType → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m,k]) → Values α .real (batch ++ [k,n]) →
    Values α .real (batch ++ [m,n])
  dotOptions : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m,k]) → Values α .real (batch ++ [k,n]) →
    Values α .real (batch ++ [m,n])
  typedDot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch ++ [m,k]) → Values α right.toTileDType (batch ++ [k,n]) →
    Values α out.toTileDType (batch ++ [m,n])
  typedSum : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (ax : Fin shape.length) → (keep : Bool) → Values α dtype.toTileDType shape →
    Values α dtype.toTileDType (TileShape.reduceShape shape ax keep)
  typedMax : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (ax : Fin shape.length) → (keep : Bool) → Values α dtype.toTileDType shape →
    Values α dtype.toTileDType (TileShape.reduceShape shape ax keep)

def Ops.algebra {α : Type} (o : Ops α) : Algebra α :=
  { o.scalar with
    compareEq := fun p d => some (o.eq p d)
    dot := fun p {_} {_} {_} {_} a b => some (o.dot p a b)
    dotWithOptions := fun p cfg {_} {_} {_} {_} a b => some (o.dotOptions p cfg a b)
    dotTyped := fun p l r out cfg {_} {_} {_} {_} a b => some (o.typedDot p l r out cfg a b)
    reduceSumTyped := fun p d {_} ax keep v => some (o.typedSum p d ax keep v)
    reduceMaxTyped := fun p d {_} ax keep v => some (o.typedMax p d ax keep v) }
end
end VeriTile.Bench.Repairs.ContextAttnMistral.FPBackend
