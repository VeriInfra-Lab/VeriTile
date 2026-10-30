import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_Binary16
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled.HalfFMA
open VeriTile.Triton
open MatmulDequantInt4.Binary16
noncomputable section

private def signed (s : Bool) (x : Rat) : Rat := if s then -x else x

private def infiniteProduct (negative : Bool) (z : FP.Decoded) : Value :=
  match z with
  | .nan .. => Value.qNaN
  | .infinity sz => if sz == negative then Value.infinity negative else Value.qNaN
  | .finite .. => Value.infinity negative

/-- Total binary16 software FMA: prepare each input, combine the exact product
with the addend, round once directly to binary16. NaNs are canonicalized;
0*Inf and opposite infinite sums are invalid. This is a chosen software
instruction profile, not an assertion about PTX NaN payload or hardware FTZ. -/
def fma (cfg : FP.Config) (x y z : Value) : Value :=
  match (prepare cfg x).decode, (prepare cfg y).decode, (prepare cfg z).decode with
  | .nan .., _, _ | _, .nan .., _ | _, _, .nan .. => Value.qNaN
  | .infinity sx, .infinity sy, z => infiniteProduct (sx != sy) z
  | .infinity sx, .finite sy y, z =>
      if y == 0 then Value.qNaN else infiniteProduct (sx != sy) z
  | .finite sx x, .infinity sy, z =>
      if x == 0 then Value.qNaN else infiniteProduct (sx != sy) z
  | .finite _ _, .finite _ _, .infinity sz => Value.infinity sz
  | .finite sx x, .finite sy y, .finite sz z =>
      let sp := sx != sy
      let product := x*y
      let negativeZero := if product == 0 && z == 0 && sp == sz then sp
        else cfg.rounding == .towardNegative
      round cfg (signed sp product + signed sz z) negativeZero

/-- The zero product is rounded before either compiled graph consumes it. -/
def zeroProduct (zero : Int) (scale : Value) : Value :=
  mul .ieeeValue (round .ieeeValue (zero : Rat)) scale

def separate (weight zero : Int) (scale : Value) : Value :=
  sub .ieeeValue (mul .ieeeValue (round .ieeeValue (weight : Rat)) scale)
    (zeroProduct zero scale)

def fused (weight zero : Int) (scale : Value) : Value :=
  fma .ieeeValue (round .ieeeValue (weight : Rat)) scale (zeroProduct zero scale).neg

theorem finite_fusion_difference :
    separate 3 2 (Value.ofBits 0x3c01) = Value.ofBits 0x3c02 ∧
    fused 3 2 (Value.ofBits 0x3c01) = Value.ofBits 0x3c01 := by decide +kernel

theorem no_universal_separate_fused_law :
    ¬ (∀ weight zero : Int, ∀ scale : Value,
      0 ≤ weight → weight ≤ 15 → 0 ≤ zero → zero ≤ 15 →
      separate weight zero scale = fused weight zero scale) := by
  intro h
  have hx := h 3 2 (Value.ofBits 0x3c01) (by omega) (by omega) (by omega) (by omega)
  rw [finite_fusion_difference.1,finite_fusion_difference.2] at hx
  cases hx

theorem special_value_vectors :
    fma .ieeeValue Value.infinity Value.zero (Value.ofBits 0x3c00) = Value.qNaN ∧
    fma .ieeeValue Value.infinity (Value.ofBits 0x3c00) (Value.infinity true) = Value.qNaN ∧
    fma .ieeeValue (Value.ofBits 0x7c01) Value.zero Value.zero = Value.qNaN ∧
    fma .ieeeValue (Value.zero true) (Value.ofBits 0x3c00) (Value.zero true) = Value.zero true ∧
    fma .ieeeValue (Value.zero true) (Value.ofBits 0x3c00) Value.zero = Value.zero ∧
    fma .ieeeValue (Value.ofBits 0x0001) (Value.ofBits 0x3c00) Value.zero = Value.ofBits 0x0001 := by
  decide +kernel

#axiomsClean finite_fusion_difference
#axiomsClean no_universal_separate_fused_law
#axiomsClean special_value_vectors
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled.HalfFMA
