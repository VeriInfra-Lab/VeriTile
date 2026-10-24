import VeriTile.Triton.Float.Structural
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.MatmulReal
open VeriTile.Triton FP.Structural
noncomputable section

/-- Real interpretation for the matrix-product/activation proofs. The actual
source uses literal, load, binary arithmetic, comparisons supplied by its dot
adapter, and the output cast. The unused reduction and bit-pattern callbacks
below carry no floating-point or reduction correctness claim. -/
def algebra (roundHalf : ℝ → ℝ) : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => Real.rpow x y
  unary := fun _ _ x => x
  cast := fun _ _ target x => if target=.fp16 then roundHalf x else x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceSum := fun _ _ _ _ _ => 0
  reduceMax := fun _ _ _ _ _ => 0

end
end VeriTile.Bench.MatmulReal
