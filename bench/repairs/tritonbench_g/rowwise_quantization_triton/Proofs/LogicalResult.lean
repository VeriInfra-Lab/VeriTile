import bench.repairs.tritonbench_g.rowwise_quantization_triton.IO
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem logical_row (p : Launch) (raw : ChannelValues ℝ (IO.inputs p)) (row : Fin p.rows) :
    IO.values p raw row.val = IO.extendRow ((IO.logicalInputs p) raw row) := by
  funext col
  simp only [IO.values,IO.extendRow,row.isLt,↓reduceDIte]
  rfl

/-- Complete typed integer-observer judgment equivalence. This retains low-byte
observation, the maximum channel, source casts, every scheduled row and frame. -/
theorem logical_real_iff (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p) :
    (IO.logicalIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule ⊨ fun xs =>
      (⟨(fun row col => ((mathematicalByte (IO.extendRow (xs row)) p.columns col.val).val : Int)),
        (fun row => logicalMaximum (IO.extendRow (xs row)) p.columns)⟩ : IO.LogicalOutputs ℝ p)) ↔
    (IO.quantizerIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule).ObservesInteger (IO.byteView p)
        (realTarget p) := by
  have target_eq : (IO.logicalIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule).lift (fun xs =>
      (⟨(fun row col => ((mathematicalByte (IO.extendRow (xs row)) p.columns col.val).val : Int)),
        (fun row => logicalMaximum (IO.extendRow (xs row)) p.columns)⟩ : IO.LogicalOutputs ℝ p)) =
      realTarget p := by
    funext raw port j
    fin_cases port
    · change ((mathematicalByte (IO.extendRow ((IO.logicalInputs p) raw j.1)) p.columns j.2.1.val).val : Int) =
        ((mathematicalByte (IO.values p raw j.1.val) p.columns j.2.1.val).val : Int)
      rw [logical_row]
    · change logicalMaximum (IO.extendRow ((IO.logicalInputs p) raw j.1)) p.columns =
        logicalMaximum (IO.values p raw j.1.val) p.columns
      rw [logical_row]
  unfold LogicalIO.Implements
  rw [target_eq]

#axiomsClean logical_real_iff
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
