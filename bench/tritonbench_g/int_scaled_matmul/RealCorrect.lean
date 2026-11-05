import bench.tritonbench_g.int_scaled_matmul.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Correct
open VeriTile.Triton FP.Structural Revision001
open scoped LogicalProgramIO BigOperators
noncomputable section
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩

def signedDot {M N K : Nat} (a : Fin M → Fin K → Int) (b : Fin K → Fin N → Int) (row : Fin M) (col : Fin N) : Int :=
  (BitVec.ofInt 32 (∑ k : Fin K, (BitVec.ofInt 8 (a row k)).toInt * (BitVec.ofInt 8 (b k col)).toInt)).toInt

def integerFormula (p : BlockWord.Parameters) (xs : LogicalIO.IntegerInputs p) (row : Fin p.rows) (col : Fin p.columns) : Int :=
  signedDot xs.a xs.b row col

def scaledFormula (p : ScaledWord.Parameters) (xs : LogicalIO.ScaledInputs p) (row : Fin p.rows) (col : Fin p.columns) : Int :=
  let x : ℝ := (signedDot xs.a xs.b row col : ℝ)*xs.rowScale row
  (BitVec.ofInt 32 (if x<0 then Int.ceil x else Int.floor x)).toInt

/-- Full grouped signed8 matrix product with the original signed32 wrap, source safety and frame. -/
specification int_matmul_correct (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    VeriTile.Spec.Real (LogicalIO.integerIO rest p ⊨ integerFormula p) :=
  (LogicalReader.integer_full_iff rest p).mpr (LogicalReader.int_matmul_correct rest p geometry positive ha hb)

/-- Ideal Real row scaling and Mathlib toward-zero truncation, over the same complete grouped grid. -/
specification int_scaled_matmul_real (rest : Algebra ℝ) (p : ScaledWord.Parameters)
    (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) :
    VeriTile.Spec.Real (LogicalIO.scaledIO rest p ⊨ scaledFormula p) :=
  (LogicalReader.scaled_full_iff rest p).mpr (LogicalReader.int_scaled_matmul_real rest p geometry even cover ha hb hs)
#axiomsClean int_matmul_correct
#axiomsClean int_scaled_matmul_real

end
end VeriTile.Bench.Repairs.IntScaledMatmul.Correct
