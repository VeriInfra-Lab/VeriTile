import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.SourceOperations
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.SourceLoad
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.SourceInteger
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPLayout
namespace VeriTile.Bench.Repairs.Fp4ToBf16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] Bits.narrow

attribute [local irreducible] shiftExpr orExpr intConst
attribute [local irreducible] loadExecution
attribute [local simp] loadExecution_mem loadExecution_x_pos loadExecution_sign_f4 loadExecution_pid loadExecution_n_elements_out
def maskExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 13 |>.take 3) (loadExecution rest p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [Kernels.source, Kernels.body, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, Kernels.body, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean maskExecution

@[local simp] theorem maskExecution_exp_biased_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_8_local_exp_biased_f4" = some (fun j => positive (laneNibble p s j) >>> 1) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_5_local_x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_4_local_sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_zero_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_6_local_zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_denormal_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_7_local_denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => (p.count : Int)*2) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (maskExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _ (by rfl)
    (maskExecution rest p s).property).trans (loadExecution_mem rest p s)
attribute [local irreducible] maskExecution
attribute [local simp] maskExecution_mem


def addBiasExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 16 |>.take 1) (maskExecution rest p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [Kernels.source, Kernels.body, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, Kernels.body, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean addBiasExecution

@[local simp] theorem addBiasExecution_exp_biased_f32 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_9_local_exp_biased_f32" = some (fun j => addedBias (laneNibble p s j)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_5_local_x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_4_local_sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_zero_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_6_local_zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_denormal_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_7_local_denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => (p.count : Int)*2) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (addBiasExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _ (by rfl)
    (addBiasExecution rest p s).property).trans (maskExecution_mem rest p s)
attribute [local irreducible] addBiasExecution
attribute [local simp] addBiasExecution_mem


def expExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 17 |>.take 1) (addBiasExecution rest p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [exponentCode_source]
  simp [exponentCode,List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, Kernels.body, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean expExecution

@[local simp] theorem expExecution_exp_biased_f32 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .int [p.block*2] "__veritile_helper_9_local_exp_biased_f32" = some (fun j => shiftedExponent (laneNibble p s j)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_5_local_x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_4_local_sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_zero_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_6_local_zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_denormal_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_7_local_denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => (p.count : Int)*2) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (expExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _ (by rfl)
    (expExecution rest p s).property).trans (addBiasExecution_mem rest p s)
attribute [local irreducible] expExecution
attribute [local simp] expExecution_mem


def normalExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 18 |>.take 5) (expExecution rest p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [normalCode_source]
  simp [normalCode,List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, expExecution_exp_biased_f32, expExecution_x_pos, expExecution_sign_f4, expExecution_zero_mask, expExecution_denormal_mask, expExecution_pid, expExecution_n_elements_out, Kernels.body, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean normalExecution

@[local simp] theorem normalExecution_result (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .int [p.block*2] "__veritile_helper_13_local_result" = some (fun j => zeroHandled (laneNibble p s j)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_5_local_x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_4_local_sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_zero_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_6_local_zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_denormal_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_7_local_denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => (p.count : Int)*2) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (normalExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _ (by rfl)
    (normalExecution rest p s).property).trans (expExecution_mem rest p s)
attribute [local irreducible] normalExecution
attribute [local simp] normalExecution_mem


def signExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 23 |>.take 3) (normalExecution rest p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [signCode_source]
  simp [signCode,List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, expExecution_exp_biased_f32, expExecution_x_pos, expExecution_sign_f4, expExecution_zero_mask, expExecution_denormal_mask, expExecution_pid, expExecution_n_elements_out, normalExecution_result, normalExecution_x_pos, normalExecution_sign_f4, normalExecution_zero_mask, normalExecution_denormal_mask, normalExecution_pid, normalExecution_n_elements_out, Kernels.body, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean signExecution

@[local simp] theorem signExecution_result (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .int [p.block*2] "__veritile_helper_13_local_result" = some (fun j => integerWord (laneNibble p s j)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_5_local_x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_4_local_sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_zero_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_6_local_zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_denormal_mask (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .bool [p.block*2] "__veritile_helper_7_local_denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => (p.count : Int)*2) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias_plain, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (signExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _ (by rfl)
    (signExecution rest p s).property).trans (normalExecution_mem rest p s)
attribute [local irreducible] signExecution
attribute [local simp] signExecution_mem


def storeExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 26 |>.take 8) (signExecution rest p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [Kernels.source, Kernels.body, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, expExecution_exp_biased_f32, expExecution_x_pos, expExecution_sign_f4, expExecution_zero_mask, expExecution_denormal_mask, expExecution_pid, expExecution_n_elements_out, normalExecution_result, normalExecution_x_pos, normalExecution_sign_f4, normalExecution_zero_mask, normalExecution_denormal_mask, normalExecution_pid, normalExecution_n_elements_out, signExecution_result, signExecution_x_pos, signExecution_sign_f4, signExecution_zero_mask, signExecution_denormal_mask, signExecution_pid, signExecution_n_elements_out, Kernels.body, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean storeExecution

/-- The complete lowered 34-statement source, including the real helper argument and return copies; no bit operation or cast is dropped. -/
def execution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.exec (backend rest) (Kernels.source p) s = some t} := by
  refine ⟨(storeExecution rest p s).val, ?_⟩
  change FP.Structural.run _
    (((Kernels.source p).surfaceBody.take 13) ++
      ((Kernels.source p).surfaceBody.drop 13 |>.take 3) ++
      ((Kernels.source p).surfaceBody.drop 16 |>.take 1) ++
      ((Kernels.source p).surfaceBody.drop 17 |>.take 1) ++
      ((Kernels.source p).surfaceBody.drop 18 |>.take 5) ++
      ((Kernels.source p).surfaceBody.drop 23 |>.take 3) ++
      ((Kernels.source p).surfaceBody.drop 26 |>.take 8)) s = _
  have hload : run (backend rest) ((Kernels.source p).surfaceBody.take 13) s = some (loadExecution rest p s).val := by
    simpa only [List.drop_zero] using (loadExecution rest p s).property
  simp only [run_append, hload,
    (maskExecution rest p s).property,
    (addBiasExecution rest p s).property, (expExecution rest p s).property,
    (normalExecution rest p s).property, (signExecution rest p s).property,
    (storeExecution rest p s).property, Option.bind_some]
#axiomsClean execution

end
end VeriTile.Bench.Repairs.Fp4ToBf16.Source
