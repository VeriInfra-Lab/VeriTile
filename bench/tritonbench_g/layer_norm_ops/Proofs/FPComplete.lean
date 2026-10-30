import bench.tritonbench_g.layer_norm_ops.Proofs.FPModel
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem bop_vector {A B C : Type} (f : A → B → C)
    (a : TileIndex [n] → A) (b : TileIndex [n] → B) :
    bop f Broadcast.nil.consSame a b = (fun i => f (a i) (b i)) := by
  funext ⟨i,⟨⟩⟩
  rfl
private theorem bop_scalarR {A B C : Type} (f : A → B → C)
    (a : TileIndex [n] → A) (b : TileIndex [] → B) :
    bop f Broadcast.scalarR a b = (fun i => f (a i) (b PUnit.unit)) := by
  funext ⟨i,⟨⟩⟩
  rfl

theorem rms_row_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (rmsKernel p) s=some t ∧ t.mem=(rowReference M p s Bool.false).mem := by
  simp [FP.Structural.exec, rmsKernel, rmsRow, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean rms_row_run

theorem plain_row_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (plainKernel p) s=some t ∧ t.mem=(rowReference M p s Bool.true).mem := by
  simp [FP.Structural.exec, plainKernel, plainRow, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean plain_row_run

theorem rms_dx_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (rmsDXKernel p) s=some t ∧ t.mem=(dxReference M p s Bool.false).mem := by
  simp [FP.Structural.exec, rmsDXKernel, rmsDX, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean rms_dx_run

theorem plain_dx_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (plainDXKernel p) s=some t ∧ t.mem=(dxReference M p s Bool.true).mem := by
  simp [FP.Structural.exec, plainDXKernel, plainDX, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean plain_dx_run

theorem residual_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (residualKernel p) s=some t ∧ t.mem=(residualReference M p s).mem := by
  simp [FP.Structural.exec, residualKernel, residualAdd, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean residual_run

theorem c1_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (c1Kernel p) s=some t ∧ t.mem=(s.write p.c1 (s.pids 0) (stored M (c1Value M p s))).mem := by
  simp [FP.Structural.exec, c1Kernel, c1Stage, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean c1_run

theorem c2_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (c2Kernel p) s=some t ∧ t.mem=(s.write p.c2 (s.pids 0) (stored M (rowC2 M p s))).mem := by
  simp [FP.Structural.exec, c2Kernel, c2Stage, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, bop_vector, bop_scalarR, numericLt, natLt, Region.cast, fold_writes_regs]
  try simp only [rowReference, dxReference, residualReference, Bool.false_eq_true,
    ↓reduceIte, scatter, fold_writes_mem, State.setReg_mem]
  simp [stored, offset, rowDX, rowDW, rowDB, rowC1, rowC2, c1Value, suppliedDX, residualValue,
    average, wdyValue, xhatValue, suppliedXhat, dyValue, scalar, weight, loaded, pad, zero, op, State.write]
  all_goals try rfl
  all_goals norm_num
  all_goals try rfl
  all_goals try (unfold wdyValue dyValue loaded pad op; norm_num; rfl)

#axiomsClean c2_run
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
