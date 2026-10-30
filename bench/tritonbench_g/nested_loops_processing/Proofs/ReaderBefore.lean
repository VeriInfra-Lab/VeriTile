import bench.tritonbench_g.nested_loops_processing.Proofs.IOImplements
import bench.tritonbench_g.nested_loops_processing.Proofs.Formula

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.ReaderBefore
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete int32 input and initial-output snapshots; explicit IO fields
and source binding are in IO.lean. Every output cell remains observed. -/
abbrev nestedIO (M : Algebra α) (rows cols : Nat) := IO.nestedIO M Bool.false rows cols

/-- The independent ordered-copy formula uses signed-int32 arange values,
zero beyond the input extent, and zero initial output; later stores win. -/
def integerTarget (rows cols : Nat) : Nat → Int :=
  let input := fun i => if i<rows*cols then (BitVec.ofNat 32 i).toInt else 0
  let inputShift := fun t =>
    let i := t/12
    let j := t%12/6
    let k := t%6/3
    if t%3=0 then 14*i else if t%3=1 then 14*i+6*j+2 else 14*i+6*j+2*k+4
  let offset := fun shift row col => ((row : Int)*(cols : Int)+((col+shift : Nat) : Int)*1).toNat
  (List.range 24).foldl (fun mem t =>
    (List.range 2).foldl (fun mem row =>
      (List.range 2).foldl (fun mem col =>
        Function.update mem (offset (2*t) row col) (input (offset (inputShift t) row col))) mem) mem) (fun _ => 0)


/-- Real is the exact signed-integer interpretation for this copy-only source.
No floating operation or numerical approximation is present. -/
theorem kernel_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (rows cols : Nat) (positive : 0 < rows*cols) :
    VeriTile.Spec.Real ((nestedIO M rows cols).ImplementsOnCells (IO.initialized rows cols)
      (IO.result rows cols (integerTarget rows cols))) := by
  have formula : integerTarget rows cols =
      target 0 (rows*cols) cols (fun i => signedValue (arangeWord i)) (fun _ => 0) := rfl
  simpa only [formula,← Formula.integer_word,Formula.integer] using
    IOProof.implements M Bool.false rows cols positive

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.ReaderBefore
