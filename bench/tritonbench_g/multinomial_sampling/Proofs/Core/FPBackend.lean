import bench.tritonbench_g.multinomial_sampling.Proofs.Core.FPSource
namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open VeriTile.Triton FP.Structural

/-- No algebraic laws: casts, scan and maximum retain their complete source
configuration; the scalar arithmetic/comparisons operate on the fp32 channels
established by the explicit score conversion and RNG. -/
structure Primitives (α : Type) where
  zero : Option ComputeDType → FloatDType → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  bits32 : Float32Bits → α
  add : α → α → α
  sub : α → α → α
  lt : α → α → Bool
  le : α → α → Bool
  scan : Option ComputeDType → {shape : TileShape} → ScanOp →
    (ax : Fin shape.length) → ScanDirection → Values α .real shape → Values α .real shape
  maximum : Option ComputeDType → {shape : TileShape} →
    (ax : Fin shape.length) → (keepDims : Bool) → Values α .real shape →
      Values α .real (TileShape.reduceShape shape ax keepDims)

noncomputable def backend {α : Type} (rest : Algebra α) (P : Primitives α) : Algebra α :=
  { rest with
    literal := fun p d _ => P.zero p d
    cast := P.cast
    fp32Bits := P.bits32
    binary := fun p d op x y => match op with
      | .add => P.add x y | .sub => P.sub x y | _ => rest.binary p d op x y
    compareLt := fun _ _ => some P.lt
    compareLe := fun _ _ => some P.le
    scan := fun p {_} op ax dir v => some (P.scan p op ax dir v)
    reduceMax := P.maximum }

end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
