import bench.tritonbench_g.rotary_emb.Proofs.ExecutionQ
namespace VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FullSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- The Q stores retain the index registers consumed by the K phase. -/
theorem qstores_seq_range (flip : Bool) :
    (qstores_run flip).val.regs .nat [2] "cur_seq_range" =
      (prefix_run flip).val.regs .nat [2] "cur_seq_range" := by
  cases flip <;> rfl

theorem qstores_head_range (flip : Bool) :
    (qstores_run flip).val.regs .nat [2] "cur_head_range" =
      (prefix_run flip).val.regs .nat [2] "cur_head_range" := by
  cases flip <;> rfl

theorem qstores_even_range (flip : Bool) :
    (qstores_run flip).val.regs .nat [2] "dim_range0" =
      (prefix_run flip).val.regs .nat [2] "dim_range0" := by
  cases flip <;> rfl

theorem qstores_odd_range (flip : Bool) :
    (qstores_run flip).val.regs .nat [2] "dim_range1" =
      (prefix_run flip).val.regs .nat [2] "dim_range1" := by
  cases flip <;> rfl

private theorem mem_ite (p : Prop) [Decidable p]
    (s t : State (WithBot ℝ)) (r : RegionName) (o : Nat) :
    (if p then s else t).mem r o = if p then s.mem r o else t.mem r o := by
  by_cases h : p <;> simp [h]

/-- Q stores leave K and both trigonometric regions unchanged. -/
theorem qstores_mem_other (flip : Bool) (r : RegionName) (o : Nat)
    (hr : r ≠ "Q") :
    (qstores_run flip).val.mem r o = (initial flip).mem r o := by
  cases flip
  all_goals simp (maxSteps := 200000) [qstores_run, qarithmetic_run, qloads_run,
    addresses_run, prefix_run, FP.Structural.store, State.write,
    State.setReg, Region.cast, TileShape.allIndices, List.finRange_succ,
    List.foldl, bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar,
    -Tile.scalar_eta, Function.comp_def, mem_ite, hr]
  all_goals simp [initial, hr, ofFloat]

def kaddresses_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 20 |>.take 4)
      (qstores_run flip).val = some t} := by
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
      qstores_seq_range, qstores_head_range, qstores_even_range, qstores_odd_range,
      qstores_mem_other, prefix_run]
    try rfl



def kloads_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 24 |>.take 6)
      (kaddresses_run flip).val = some t} := by
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
      qstores_seq_range, qstores_head_range, qstores_even_range, qstores_odd_range,
      qstores_mem_other, prefix_run, kaddresses_run]
    try rfl



def karithmetic_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 30 |>.take 2)
      (kloads_run flip).val = some t} := by
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
      qstores_seq_range, qstores_head_range, qstores_even_range, qstores_odd_range,
      qstores_mem_other, prefix_run, kaddresses_run, kloads_run]
    try rfl



def kstores_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      ((program Bool.false flip).surfaceBody.drop 32 |>.take 2)
      (karithmetic_run flip).val = some t} := by
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
      qstores_seq_range, qstores_head_range, qstores_even_range, qstores_odd_range,
      qstores_mem_other, prefix_run, kaddresses_run, kloads_run, karithmetic_run]
    try rfl


end
end VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
