import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- A reader's ordinary finite maximum recurrence determines exactly the old
mathematical maximum, hence the full original typed integer-observer judgment. -/
theorem reader_judgment_iff (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p)
    (maximumAbs : (Nat → ℝ) → Nat → ℝ)
    (zero : ∀ values, maximumAbs values 0=0)
    (step : ∀ values n, maximumAbs values (n+1)=max (maximumAbs values n) |values n|) :
    (IO.logicalIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule ⊨ fun xs =>
      ({ quantized := fun row col =>
          let m := maximumAbs (IO.extendRow (xs row)) p.columns
          ((if m=0 then narrow8 0 else narrow8 (roundEven (127*(xs row col/m)))).val : Int)
         maximum := fun row => maximumAbs (IO.extendRow (xs row)) p.columns } : IO.LogicalOutputs ℝ p)) ↔
    (IO.quantizerIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule).ObservesInteger (IO.byteView p)
        (realTarget p) := by
  have maximumAbs_eq (values : Nat → ℝ) (n : Nat) :
      maximumAbs values n = logicalMaximum values n := by
    induction n with
    | zero => exact zero values
    | succ n ih => simp only [step,logicalMaximum,ih]
  have connection : (fun xs : Fin p.rows → Fin p.columns → ℝ =>
      ({ quantized := fun row col =>
          let m := maximumAbs (IO.extendRow (xs row)) p.columns
          ((if m=0 then narrow8 0 else narrow8 (roundEven (127*(xs row col/m)))).val : Int)
         maximum := fun row => maximumAbs (IO.extendRow (xs row)) p.columns } : IO.LogicalOutputs ℝ p)) =
      (fun xs : Fin p.rows → Fin p.columns → ℝ => (⟨(fun row col => ((mathematicalByte (IO.extendRow (xs row)) p.columns col.val).val : Int)),
        (fun row => logicalMaximum (IO.extendRow (xs row)) p.columns)⟩ : IO.LogicalOutputs ℝ p)) := by
    funext xs
    simp only [maximumAbs_eq,mathematicalByte,IO.extendRow,Fin.isLt,↓reduceDIte]
  rw [connection]
  exact logical_real_iff rest p schedule

theorem reader_correct (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p)
    (maximumAbs : (Nat → ℝ) → Nat → ℝ)
    (zero : ∀ values, maximumAbs values 0=0)
    (step : ∀ values n, maximumAbs values (n+1)=max (maximumAbs values n) |values n|) :
    (IO.logicalIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule ⊨ fun xs =>
      ({ quantized := fun row col =>
          let m := maximumAbs (IO.extendRow (xs row)) p.columns
          ((if m=0 then narrow8 0 else narrow8 (roundEven (127*(xs row col/m)))).val : Int)
         maximum := fun row => maximumAbs (IO.extendRow (xs row)) p.columns } : IO.LogicalOutputs ℝ p)) :=
  (reader_judgment_iff rest p schedule maximumAbs zero step).mpr (real rest p schedule)

#axiomsClean reader_judgment_iff
#axiomsClean reader_correct
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
