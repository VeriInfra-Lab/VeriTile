import bench.tritonbench_g.rotary_emb.Proofs.ExecutionModel

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FullSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- Actual first six statements, retaining both program IDs and all ranges. -/
def prefix_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.take 6) (initial flip) = some t} := by
  cases flip
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 200000) [program, kernel, original, template,
      template_fp16_bf16, template_fp64_bf16, template_fp64_fp32, template_fp16_fp32,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, bind, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl



def addresses_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 6 |>.take 4)
      (prefix_run flip).val = some t} := by
  cases flip
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 200000) [program, kernel, original, template,
      template_fp16_bf16, template_fp64_bf16, template_fp64_fp32, template_fp16_fp32,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, bind, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
      prefix_run]
    try rfl


def qloads_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 10 |>.take 6)
      (addresses_run flip).val = some t} := by
  cases flip
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 200000) [program, kernel, original, template,
      template_fp16_bf16, template_fp64_bf16, template_fp64_fp32, template_fp16_fp32,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, bind, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
      prefix_run, addresses_run]
    try rfl


def qarithmetic_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 16 |>.take 2)
      (qloads_run flip).val = some t} := by
  cases flip
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 200000) [program, kernel, original, template,
      template_fp16_bf16, template_fp64_bf16, template_fp64_fp32, template_fp16_fp32,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, bind, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
      prefix_run, addresses_run, qloads_run]
    try rfl


def qstores_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 18 |>.take 2)
      (qarithmetic_run flip).val = some t} := by
  cases flip
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 200000) [program, kernel, original, template,
      template_fp16_bf16, template_fp64_bf16, template_fp64_fp32, template_fp16_fp32,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, bind, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
      prefix_run, addresses_run, qloads_run, qarithmetic_run]
    try rfl

end
end VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
