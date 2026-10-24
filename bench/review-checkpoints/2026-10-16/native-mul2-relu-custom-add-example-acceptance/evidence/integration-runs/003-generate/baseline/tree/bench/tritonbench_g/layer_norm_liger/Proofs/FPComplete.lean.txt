import bench.tritonbench_g.layer_norm_liger.Proofs.FPModel
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem complete_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (withSum M R) (kernel p) s = some t ∧
      t.mem = (reference M R p s).mem := by
  simp [FP.Structural.exec, kernel, source, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, withSum]
  simp only [reference, fold_writes_mem, State.setReg_mem]
  simp [stored, meanOffset, rstdOffset, outputOffset, yValue, rstdValue, varianceValue,
    centered, meanValue, denominator, xValues, wValues, bValues, input, padding, State.write]
  rfl

#axiomsClean complete_run
end
end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
