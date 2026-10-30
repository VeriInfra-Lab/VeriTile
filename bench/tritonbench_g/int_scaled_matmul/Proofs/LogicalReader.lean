import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.int_scaled_matmul.IO
import bench.tritonbench_g.int_scaled_matmul.Proofs.IOComplete
import bench.tritonbench_g.int_scaled_matmul.Proofs.RealMath
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.IntScaledMatmul.LogicalReader
open VeriTile.Triton FP.Structural Revision001
open scoped ProgramIO
noncomputable section
local instance wordDefault : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩

/-- Full logical K sum with the source's signed operand and result widths. -/
def signedDot {M N K : Nat} (a : TileIndex [M,K] → Int)
    (b : TileIndex [K,N] → Int) (i : TileIndex [M,N]) : Int :=
  (BitVec.ofInt 32 (LogicalSum.sum (fun k => if hk : k<K then
    (BitVec.ofInt 8 (a (i.1,⟨k,hk⟩,PUnit.unit))).toInt *
      (BitVec.ofInt 8 (b (⟨k,hk⟩,i.2.1,PUnit.unit))).toInt else 0) K)).toInt

def integerFormula (p : BlockWord.Parameters)
    (vals : ChannelValues KernelModel.Word (IntIO.inputs p)) :
    ChannelValues KernelModel.Word (IntIO.outputs p) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact signedDot (vals (0 : Fin 2)) (vals (1 : Fin 2))

/-- Row scaling uses Real multiplication and toward-zero floor/ceiling,
followed by the existing signed32 result projection. -/
def scaledFormula (p : ScaledWord.Parameters)
    (vals : ChannelValues ℝ (RealIO.inputs p)) : ChannelValues ℝ (RealIO.outputs p) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact fun j =>
    let scale : ℝ := vals (2 : Fin 3) (j.1,PUnit.unit)
    let x : ℝ := (signedDot (vals (0 : Fin 3)) (vals (1 : Fin 3)) j : ℝ) *
      scale
    (BitVec.ofInt 32 (if x<0 then Int.ceil x else Int.floor x)).toInt


/-- Complete signed8 matrix product with the original signed32 wrap. Shared
IO proves actual source access safety, successful full-grid execution, all
logical outputs and the entire exterior memory frame. -/
theorem int_matmul_correct (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    VeriTile.Spec.Real (IntIO.io rest p ⊨ integerFormula p) :=
  IntIO.complete rest p geometry positive ha hb

/-- The same scaled source interpreted with ordinary Real multiplication,
exact integer-to-Real promotion, Mathlib toward-zero truncation and the explicit
signed32 result projection. This ideal-Real theorem does not assert native
FP32 equality or restrict the all-word FP pair. The full grouped grid, both
EVEN_K branches, empty outputs, K=0, access safety and exterior frame remain. -/
theorem int_scaled_matmul_real (rest : Algebra ℝ) (p : ScaledWord.Parameters)
    (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) :
    VeriTile.Spec.Real (RealIO.io rest p ⊨ scaledFormula p) :=
  by
    have target : scaledFormula p = RealIO.formula p := by
      funext vals port
      have hp : port=(0 : Fin 1) := Fin.eq_zero port
      subst port
      funext j
      rfl
    simpa only [target] using RealIO.complete rest p geometry even cover ha hb hs

end
end VeriTile.Bench.Repairs.IntScaledMatmul.LogicalReader

namespace VeriTile.Bench.Repairs.IntScaledMatmul.LogicalReader
open VeriTile.Triton FP.Structural Revision001
open scoped LogicalProgramIO BigOperators
noncomputable section
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩

def logicalDot {M N K : Nat} (a : Fin M → Fin K → Int) (b : Fin K → Fin N → Int) (row : Fin M) (col : Fin N) : Int :=
  (BitVec.ofInt 32 (∑ k : Fin K, (BitVec.ofInt 8 (a row k)).toInt * (BitVec.ofInt 8 (b k col)).toInt)).toInt

def logicalIntegerFormula (p : BlockWord.Parameters) (xs : LogicalIO.IntegerInputs p) (row : Fin p.rows) (col : Fin p.columns) : Int :=
  logicalDot xs.a xs.b row col

def logicalScaledFormula (p : ScaledWord.Parameters) (xs : LogicalIO.ScaledInputs p) (row : Fin p.rows) (col : Fin p.columns) : Int :=
  let x : ℝ := (logicalDot xs.a xs.b row col : ℝ)*xs.rowScale row
  (BitVec.ofInt 32 (if x<0 then Int.ceil x else Int.floor x)).toInt

theorem sum_eq_fin (f : Nat → Int) (n : Nat) : LogicalSum.sum f n = ∑ k : Fin n, f k.val := by
  induction n with
  | zero => simp [LogicalSum.sum]
  | succ n ih => simpa only [LogicalSum.sum,Fin.sum_univ_castSucc,Fin.coe_castSucc,Fin.val_last] using congrArg (fun x => x+f n) ih

theorem dot_curried {M N K : Nat} (a : TileIndex [M,K] → Int) (b : TileIndex [K,N] → Int) (j : TileIndex [M,N]) :
    signedDot a b j = logicalDot (fun r c => a (r,c,())) (fun r c => b (r,c,())) j.1 j.2.1 := by
  unfold signedDot logicalDot
  rw [sum_eq_fin]
  simp only [Fin.isLt,dif_pos]

theorem integer_full_iff (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) :
    (LogicalIO.integerIO rest p ⊨ logicalIntegerFormula p) ↔ (IntIO.io rest p).Implements (integerFormula p) := by
  have target : (LogicalIO.integerIO rest p).lift (logicalIntegerFormula p) = integerFormula p := by
    funext xs o; fin_cases o; funext j
    change logicalDot (fun r c => xs (0 : Fin 2) (r,c,())) (fun r c => xs (1 : Fin 2) (r,c,())) j.1 j.2.1 =
      signedDot (xs (0 : Fin 2)) (xs (1 : Fin 2)) j
    exact (dot_curried _ _ _).symm
  change (IntIO.io rest p).Implements _ ↔ _
  simpa only [target]

theorem scaled_full_iff (rest : Algebra ℝ) (p : ScaledWord.Parameters) :
    (LogicalIO.scaledIO rest p ⊨ logicalScaledFormula p) ↔ (RealIO.io rest p).Implements (scaledFormula p) := by
  have target : (LogicalIO.scaledIO rest p).lift (logicalScaledFormula p) = scaledFormula p := by
    funext xs o; fin_cases o; funext j
    exact congrArg (fun acc : Int => RealIO.resultValue acc (((LogicalIO.scaledInputs p) xs).rowScale j.1))
      (dot_curried (xs (0 : Fin 3)) (xs (1 : Fin 3)) j).symm
  change (RealIO.io rest p).Implements _ ↔ _
  simpa only [target]

end
end VeriTile.Bench.Repairs.IntScaledMatmul.LogicalReader
