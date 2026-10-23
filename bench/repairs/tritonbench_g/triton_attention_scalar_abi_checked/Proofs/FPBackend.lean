import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPSource
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural

/-- Explicit operation interface, with no numerical identities. In particular,
half dot and half additions are separate operations in this source graph. -/
structure Primitives (α : Type) where
  literal : Option ComputeDType → FloatDType → ℝ → α
  negInf : α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  unary : Option ComputeDType → Unary → α → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  max : Option ComputeDType → {shape : TileShape} → (ax : Fin shape.length) →
    (keep : Bool) → Values α .real shape → Values α .real (TileShape.reduceShape shape ax keep)
  sum : Option ComputeDType → {shape : TileShape} → (ax : Fin shape.length) →
    (keep : Bool) → Values α .real shape → Values α .real (TileShape.reduceShape shape ax keep)
  dot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) → Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

noncomputable def backend {α : Type} (rest : Algebra α) (P : Primitives α) : Algebra α :=
  { rest with
    literal := P.literal
    negInf := P.negInf
    binary := P.binary
    unary := P.unary
    cast := P.cast
    reduceMax := P.max
    reduceSum := P.sum
    dotTyped := fun p left right out opts {_} {_} {_} {_} a b => some (P.dot p left right out opts a b) }
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
