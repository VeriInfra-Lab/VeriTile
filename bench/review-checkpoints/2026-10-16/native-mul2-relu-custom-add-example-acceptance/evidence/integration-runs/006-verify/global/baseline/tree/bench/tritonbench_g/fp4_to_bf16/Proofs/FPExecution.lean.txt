import bench.tritonbench_g.fp4_to_bf16.Proofs.FPInteger
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPLayout
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] Bits.narrow

def loadExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((source p).surfaceBody.drop 0 |>.take 11) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, List.drop, List.take, unscaled, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean loadExecution

@[local simp] theorem loadExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .nat [p.block*2] "x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right]
  try rfl

@[local simp] theorem loadExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .nat [p.block*2] "sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right]
  try rfl

@[local simp] theorem loadExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right]
  try rfl

@[local simp] theorem loadExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right]
  try rfl

@[local simp] theorem loadExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.mem = s.mem := by simp [loadExecution]
attribute [local irreducible] loadExecution


end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
