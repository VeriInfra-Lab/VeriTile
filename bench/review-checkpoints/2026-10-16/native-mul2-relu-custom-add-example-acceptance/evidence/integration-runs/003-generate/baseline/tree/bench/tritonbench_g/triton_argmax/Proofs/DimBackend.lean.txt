import bench.tritonbench_g.triton_argmax.Proofs.DimSource
import bench.tritonbench_g.matrix_reduction.Proofs.ConcreteCorrect
import bench.tritonbench_g.triton_argmax.Proofs.BitCorrect
import VeriTile.Triton.Float.Control
namespace VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend
open VeriTile.Triton FP.Structural TypedSource MatrixReduction.MachineMax
open MatrixReduction.ConcreteCorrect (Word)

/-- Normalize a floating channel to its actual binary32 comparison encoding.
Indexed max still returns the original storage-channel word. -/
def normalize (dtype : FloatDType) (x : Word) : Word :=
  match dtype with
  | .fp16 => MatrixReduction.BinaryConversion.widen (x.truncate 16)
  | _ => x

def indexed (dtype : FloatDType) {shape : TileShape}
    (ax : Fin shape.length) (xs : Values Word dtype.toTileDType shape) :
    Option (Values Word dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values Word .nat (TileShape.eraseAxis shape ax)) :=
  if hn : 0 < TileShape.axisDim shape ax then
    let pick := fun out => best .fp32
      (fun k => normalize dtype (toFloat dtype (xs (TileShape.insertAxisIndex shape ax out k)))) hn
    some ((fun out => xs (TileShape.insertAxisIndex shape ax out (pick out))),
      fun out => (pick out).val)
  else none

def backend (rest : Algebra Word) : Algebra Word :=
  { MatrixReduction.ConcreteCorrect.backend rest with
    negInf := negativeInfinity .fp32
    indexedMax := fun _ dtype => indexed dtype
    compareLt := fun _ dtype => some (fun x y =>
      decide (rank .fp32 (normalize dtype x) < rank .fp32 (normalize dtype y))) }

/-- Ordered non-NaN machine values are bounded below by negative infinity. -/
theorem negativeInfinity_le (x : Bits .fp32) (hx : isNaN .fp32 x = Bool.false) :
    rank .fp32 (negativeInfinity .fp32) ≤ rank .fp32 x := by
  have h : x.toNat < 4294967296 := x.isLt
  simp only [isNaN, magnitude, width, fractionBits, exponentBits] at hx
  norm_num at hx
  change 8388607 ≤ (MatrixReduction.BinaryConversion.key32 x).toNat
  rw [MatrixReduction.BinaryConversion.key32_toNat]
  split_ifs <;> omega

#axiomsClean negativeInfinity_le
end VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend
