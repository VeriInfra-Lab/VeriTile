import bench.tritonbench_g.rowwise_quantization_triton.Proofs.ReaderResult

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Specification
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Maximum absolute value over exactly the logical row, with empty maximum
zero. This is a pure finite maximum, independent of the reduction tree. -/
def maximumAbs (values : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => max (maximumAbs values n) |values n|

/-- Both typed outputs: nearest-even low bytes with a zero-row branch, and the
mathematical maximum. Real erases only the FP16 float cast. -/
def formula (p : Launch) (xs : Fin p.rows → Fin p.columns → ℝ) : IO.LogicalOutputs ℝ p where
  quantized := fun row col =>
    let m := maximumAbs (IO.extendRow (xs row)) p.columns
    ((if m=0 then narrow8 0 else narrow8 (roundEven (127*(xs row col/m)))).val : Int)
  maximum := fun row => maximumAbs (IO.extendRow (xs row)) p.columns

abbrev quantizerIO (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p) :=
  IO.logicalIO (KernelRealResult.algebra p rest) KernelRealResult.round
    (KernelProfile.equal KernelRealResult.ops) p schedule

/-- Complete source grid, both observable typed outputs and exact frame.
IO names source precision, llrint, integer representation and byte observation. -/
specification rowwise_grid_real (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p) :
    Spec.Real (quantizerIO rest p schedule ⊨ formula p) :=
  IOResults.reader_correct rest p schedule maximumAbs (fun _ => rfl) (fun _ _ => rfl)

#specNonCircular formula avoiding [RowwiseQuantizationTriton.source,KernelExecution.exec,KernelGrid.run]
#axiomsClean rowwise_grid_real
#stmtSurfaceSubset rowwise_grid_real ⊆
  [Spec.Real,FP.Structural.Algebra,Launch,Schedule,quantizerIO,LogicalIO.Implements,formula,ChannelValues,IO.inputs,IO.outputs,Launch.rows,Launch.columns,
   IO.LogicalOutputs,IntegerProgramIO.ObservesInteger,IO.quantizerIO,KernelRealResult.algebra,
   KernelRealResult.round,KernelProfile.equal,KernelRealResult.ops,IO.byteView]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Specification
