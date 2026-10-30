import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPDecodeOperations
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPDecodeLoad
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPLayout
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] Bits.narrow

attribute [local irreducible] shiftExpr orExpr intConst
attribute [local irreducible] loadExecution
attribute [local simp] loadExecution_mem loadExecution_x_pos loadExecution_sign_f4 loadExecution_pid loadExecution_n_elements_out loadExecution_n_elements_s
def maskExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 12 |>.take 3) (loadExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean maskExecution

@[local simp] theorem maskExecution_exp_biased_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .nat [p.block*2] "exp_biased_f4" = some (fun j => positive (laneNibble p s j) >>> 1) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_x_pos (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_sign_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_zero_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .bool [p.block*2] "zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_denormal_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .bool [p.block*2] "denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun j => p.count*2/32) := by
  simp [maskExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem maskExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (maskExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (maskExecution rest P p s).property).trans (loadExecution_mem rest P p s)
attribute [local irreducible] maskExecution
attribute [local simp] maskExecution_mem


def minusExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 15 |>.take 1) (maskExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean minusExecution

@[local simp] theorem minusExecution_exp_biased_f32 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .nat [p.block*2] "exp_biased_f32" = some (fun j => minusBias (laneNibble p s j)) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_x_pos (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_sign_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_zero_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .bool [p.block*2] "zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_denormal_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .bool [p.block*2] "denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun j => p.count*2/32) := by
  simp [minusExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem minusExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (minusExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (minusExecution rest P p s).property).trans (maskExecution_mem rest P p s)
attribute [local irreducible] minusExecution
attribute [local simp] minusExecution_mem


def addBiasExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 16 |>.take 1) (minusExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, minusExecution_exp_biased_f32, minusExecution_x_pos, minusExecution_sign_f4, minusExecution_zero_mask, minusExecution_denormal_mask, minusExecution_pid, minusExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean addBiasExecution

@[local simp] theorem addBiasExecution_exp_biased_f32 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .nat [p.block*2] "exp_biased_f32" = some (fun j => addedBias (laneNibble p s j)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_x_pos (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_sign_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_zero_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .bool [p.block*2] "zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_denormal_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .bool [p.block*2] "denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun j => p.count*2/32) := by
  simp [addBiasExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem addBiasExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (addBiasExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (addBiasExecution rest P p s).property).trans (minusExecution_mem rest P p s)
attribute [local irreducible] addBiasExecution
attribute [local simp] addBiasExecution_mem


def expExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 17 |>.take 1) (addBiasExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [exponentCode_source]
  simp [exponentCode,List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, minusExecution_exp_biased_f32, minusExecution_x_pos, minusExecution_sign_f4, minusExecution_zero_mask, minusExecution_denormal_mask, minusExecution_pid, minusExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean expExecution

@[local simp] theorem expExecution_exp_biased_f32 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .int [p.block*2] "exp_biased_f32" = some (fun j => shiftedExponent (laneNibble p s j)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_x_pos (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_sign_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_zero_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .bool [p.block*2] "zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_denormal_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .bool [p.block*2] "denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun j => p.count*2/32) := by
  simp [expExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem expExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (expExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (expExecution rest P p s).property).trans (addBiasExecution_mem rest P p s)
attribute [local irreducible] expExecution
attribute [local simp] expExecution_mem


def normalExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 18 |>.take 5) (expExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [normalCode_source]
  simp [normalCode,List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, minusExecution_exp_biased_f32, minusExecution_x_pos, minusExecution_sign_f4, minusExecution_zero_mask, minusExecution_denormal_mask, minusExecution_pid, minusExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, expExecution_exp_biased_f32, expExecution_x_pos, expExecution_sign_f4, expExecution_zero_mask, expExecution_denormal_mask, expExecution_pid, expExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean normalExecution

@[local simp] theorem normalExecution_result (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .int [p.block*2] "result" = some (fun j => zeroHandled (laneNibble p s j)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_x_pos (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_sign_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_zero_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .bool [p.block*2] "zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_denormal_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .bool [p.block*2] "denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun j => p.count*2/32) := by
  simp [normalExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem normalExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (normalExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (normalExecution rest P p s).property).trans (expExecution_mem rest P p s)
attribute [local irreducible] normalExecution
attribute [local simp] normalExecution_mem


def signExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 23 |>.take 3) (normalExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [signCode_source]
  simp [signCode,List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, minusExecution_exp_biased_f32, minusExecution_x_pos, minusExecution_sign_f4, minusExecution_zero_mask, minusExecution_denormal_mask, minusExecution_pid, minusExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, expExecution_exp_biased_f32, expExecution_x_pos, expExecution_sign_f4, expExecution_zero_mask, expExecution_denormal_mask, expExecution_pid, expExecution_n_elements_out, normalExecution_result, normalExecution_x_pos, normalExecution_sign_f4, normalExecution_zero_mask, normalExecution_denormal_mask, normalExecution_pid, normalExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean signExecution

@[local simp] theorem signExecution_result (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .int [p.block*2] "result" = some (fun j => integerWord (laneNibble p s j)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_x_pos (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_sign_f4 (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_zero_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .bool [p.block*2] "zero_mask" = some (fun j => decide (positive (laneNibble p s j)=0)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_denormal_mask (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .bool [p.block*2] "denormal_mask" = some (fun j => decide (positive (laneNibble p s j)=1)) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun j => p.count*2/32) := by
  simp [signExecution,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32, shiftedMantissa, normalWord, zeroHandled, denormalHandled, shiftedSign, integerWord, or32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits]
  try rfl

@[local simp] theorem signExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (signExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (signExecution rest P p s).property).trans (normalExecution_mem rest P p s)
attribute [local irreducible] signExecution
attribute [local simp] signExecution_mem


def decodeExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.drop 26 |>.take 2) (signExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [List.drop, List.take, loadExecution_x_pos, loadExecution_sign_f4, loadExecution_pid, loadExecution_n_elements_out, maskExecution_exp_biased_f4, maskExecution_x_pos, maskExecution_sign_f4, maskExecution_zero_mask, maskExecution_denormal_mask, maskExecution_pid, maskExecution_n_elements_out, minusExecution_exp_biased_f32, minusExecution_x_pos, minusExecution_sign_f4, minusExecution_zero_mask, minusExecution_denormal_mask, minusExecution_pid, minusExecution_n_elements_out, addBiasExecution_exp_biased_f32, addBiasExecution_x_pos, addBiasExecution_sign_f4, addBiasExecution_zero_mask, addBiasExecution_denormal_mask, addBiasExecution_pid, addBiasExecution_n_elements_out, expExecution_exp_biased_f32, expExecution_x_pos, expExecution_sign_f4, expExecution_zero_mask, expExecution_denormal_mask, expExecution_pid, expExecution_n_elements_out, normalExecution_result, normalExecution_x_pos, normalExecution_sign_f4, normalExecution_zero_mask, normalExecution_denormal_mask, normalExecution_pid, normalExecution_n_elements_out, signExecution_result, signExecution_x_pos, signExecution_sign_f4, signExecution_zero_mask, signExecution_denormal_mask, signExecution_pid, signExecution_n_elements_out, scaledCandidate, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind, Option.map,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

@[local simp] theorem decodeExecution_output (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (decodeExecution rest P p s).val.regs .bf16 [p.block*2] "output" =
      some (fun j => converted (laneNibble p s j)) := by
  simp [decodeExecution, converted, embed]
  rfl

@[local simp] theorem decodeExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (decodeExecution rest P p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest P) _ _ _ (by rfl)
    (decodeExecution rest P p s).property).trans (signExecution_mem rest P p s)

@[local simp] theorem decodeExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (decodeExecution rest P p s).val.regs .nat [] "pid" = some (fun _ => s.pids 0) := by
  simp [decodeExecution]

@[local simp] theorem decodeExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (decodeExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun _ => p.count*2) := by
  simp [decodeExecution]

@[local simp] theorem decodeExecution_n_elements_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (decodeExecution rest P p s).val.regs .nat [] "n_elements_s" = some (fun _ => p.count*2/32) := by
  simp [decodeExecution]

#axiomsClean decodeExecution

/-- The complete decoded-prefix of the scaled candidate, including the early scale count. -/
def decodedPrefix (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest P) ((source p).surfaceBody.take 28) s = some t} := by
  refine ⟨(decodeExecution rest P p s).val, ?_⟩
  change FP.Structural.run _
    (((source p).surfaceBody.take 12) ++
      ((source p).surfaceBody.drop 12 |>.take 3) ++
      ((source p).surfaceBody.drop 15 |>.take 1) ++
      ((source p).surfaceBody.drop 16 |>.take 1) ++
      ((source p).surfaceBody.drop 17 |>.take 1) ++
      ((source p).surfaceBody.drop 18 |>.take 5) ++
      ((source p).surfaceBody.drop 23 |>.take 3) ++
      ((source p).surfaceBody.drop 26 |>.take 2)) s = _
  have hload : run (backend rest P) ((source p).surfaceBody.take 12) s = some (loadExecution rest P p s).val := by
    simpa only [List.drop_zero] using (loadExecution rest P p s).property
  simp only [run_append, hload,
    (maskExecution rest P p s).property, (minusExecution rest P p s).property,
    (addBiasExecution rest P p s).property, (expExecution rest P p s).property,
    (normalExecution rest P p s).property, (signExecution rest P p s).property,
    (decodeExecution rest P p s).property, Option.bind_some]
#axiomsClean decodedPrefix

end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
