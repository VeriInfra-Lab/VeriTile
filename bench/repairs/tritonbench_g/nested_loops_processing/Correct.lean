import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section
/-- Signed-int32 arange input and zero initial output, as logical matrices. -/
def initialized (rows cols : Nat) (xs : LogicalView.Inputs rows cols) : Prop :=
  (∀ r c, xs.input r c=(BitVec.ofNat 32 (r.val*cols+c.val)).toInt) ∧
  (∀ r c, xs.initialOutput r c=0)

/-- Ordered 2×2 shifted copies. A column crossing the row width carries to
the next row. Stores are traversed in source order, so the final writer wins. -/
def integerTarget (rows cols : Nat) (row : Fin rows) (column : Fin cols) : Int :=
  let inputShift := fun t =>
    let i := t/12
    let j := t%12/6
    let k := t%6/3
    if t%3=0 then 14*i else if t%3=1 then 14*i+6*j+2 else 14*i+6*j+2*k+4
  (List.range 24).foldl (fun value t =>
    (List.range 2).foldl (fun value r =>
      (List.range 2).foldl (fun value c =>
        if row.val=r+(c+2*t)/cols ∧ column.val=(c+2*t)%cols then
          let inputRow := r+(c+inputShift t)/cols
          let inputColumn := (c+inputShift t)%cols
          if inputRow<rows then (BitVec.ofNat 32 (inputRow*cols+inputColumn)).toInt else 0
        else value) value) value) 0

/-- Exact int32 input/output cells, all source loops/barriers, source safety,
execution and full frame. Real denotes the exact copy-only integer model. -/
specification kernel_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (rows cols : Nat) (positive : 0<rows*cols) :
    VeriTile.Spec.Real (LogicalView.nestedIO M rows cols ⊨{initialized rows cols} (fun _ => integerTarget rows cols)) :=
  (LogicalReader.judgment_iff M rows cols positive).mpr (ReaderBefore.kernel_correctness M rows cols positive)
#axiomsClean kernel_correctness
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct
