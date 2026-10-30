import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MemoryHelpers

namespace VeriTile.Bench.Repairs.LayerNormOps.Forward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option pp.maxSteps 1200
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem run01 {α : Type} [Inhabited α] (M : Algebra α)
    (p : ForwardParameters) (s : State α)
    (hr : p.rms = Bool.false) (ha : p.hasResidual = Bool.true)
    :
    ∃ t, FP.Structural.exec M (forwardKernel p) s = some t ∧ t.mem = (reference M p s).mem := by
  cases hs : p.storeResidual <;> cases hb : p.hasBias
  all_goals
    simp [FP.Structural.exec, forwardKernel, forwardSource, hr, ha, hs, hb,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store,
      Option.bind, ComputeDType.eraseDType, FloatDType.toTileDType,
      FloatDType.numericDType, ofFloat, toFloat, numeric, bop, numericLt,
      natLt, Region.cast, vector_bop]
    simp only [reference, scatter, fold_writes_mem, State.setReg_mem]
    simp [statisticsState, residualState, scatter, stored, yValue, rstdValue,
      varianceValue, centered, meanValue, average, z, loaded, unspecified,
      padding, literal, binary, unhintedBinary, cast, hr, ha, hs, hb,
      fold_writes_mem, fold_writes_undef]
    try simp only [write_mem, State.write_undef, State.setReg_mem, State.setReg_undef, fold_writes_mem, fold_writes_undef, vector_bop, region_name_eq]
    try norm_num only
    try rfl
    try dsimp only [z, loaded, binary, cast, padding, literal]
    try simp only [ha, Bool.false_eq_true, if_false, State.setReg_mem, State.setReg_undef]
    try rfl
    congr! 24 <;> try simp [z, loaded, binary, cast, padding, literal, ha, State.setReg_mem, State.setReg_undef]
    all_goals try rfl
    all_goals try delta z
    all_goals try simp [ha, loaded, cast, padding, literal]
    all_goals try simp only [fold_writes_mem, fold_writes_undef, State.setReg_mem, State.setReg_undef]
    all_goals try rfl

#axiomsClean run01
end
end VeriTile.Bench.Repairs.LayerNormOps.Forward
