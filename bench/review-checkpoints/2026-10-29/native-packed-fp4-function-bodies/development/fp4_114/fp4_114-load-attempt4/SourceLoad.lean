import bench.repairs.tritonbench_g.fp4_to_bf16.Kernels
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPInteger
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

private def packedExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.take 7) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [Kernels.source,Kernels.body,ComputeKernel.surfaceBody,List.drop,List.take]
  simp [FP.Structural.exec,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl


@[local simp] private theorem packedExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (packedExecution rest p s).val.mem = s.mem := by simp [packedExecution]
@[local simp] private theorem packedExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (packedExecution rest p s).val.regs .int [] "pid" = some (fun _ => (s.pids 0 : Int)) := by
  simp [packedExecution, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
@[local simp] private theorem packedExecution_count (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (packedExecution rest p s).val.regs .int [] "n_elements_out" = some (fun _ => (p.count : Int)*2) := by
  simp [packedExecution, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
@[local simp] private theorem packedExecution_loaded (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (packedExecution rest p s).val.regs .nat [p.block] "x_packed" = some (fun j => loaded p s j.1.val) := by
  simp [packedExecution, loaded, inputOffset, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, ← Int.natCast_mul, ← Int.natCast_add]
attribute [local irreducible] packedExecution

def loadExecution (rest : Algebra Word) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.run (backend rest) ((Kernels.source p).surfaceBody.drop 0 |>.take 13) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  change run _ (((Kernels.source p).surfaceBody.take 7) ++
    ((Kernels.source p).surfaceBody.drop 7 |>.take 6)) s = _
  rw [run_append,(packedExecution rest p s).property]
  dsimp only [Option.bind_some]
  dsimp only [Kernels.source,Kernels.body,ComputeKernel.surfaceBody,List.drop,List.take]
  simp [FP.Structural.exec,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.product, ComputeDType.eraseDType,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Broadcast.leftIndex, Broadcast.rightIndex,
    reshape_pair, TileShape.splitLastIndex]
  try rfl

#axiomsClean loadExecution

@[local simp] theorem loadExecution_x_pos (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_5_local_x_pos" = some (fun j => positive (laneNibble p s j)) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right,
    ← Int.natCast_mul, ← Int.natCast_add]
  try rfl

@[local simp] theorem loadExecution_sign_f4 (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .nat [p.block*2] "__veritile_helper_4_local_sign_f4" = some (fun j => sign (laneNibble p s j)) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right,
    ← Int.natCast_mul, ← Int.natCast_add]
  try rfl

@[local simp] theorem loadExecution_pid (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right,
    ← Int.natCast_mul, ← Int.natCast_add]
  try rfl

@[local simp] theorem loadExecution_n_elements_out (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => (p.count : Int)*2) := by
  simp [loadExecution, loaded, laneNibble, inputOffset, positive, sign,
    shiftedExponent, addedBias, minusBias, signed32, unsigned32,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    IntegerExpr.same_left, IntegerExpr.same_right,
    ← Int.natCast_mul, ← Int.natCast_add]
  try rfl

@[local simp] theorem loadExecution_mem (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (loadExecution rest p s).val.mem = s.mem := by simp [loadExecution]
attribute [local irreducible] loadExecution


end
end VeriTile.Bench.Repairs.Fp4ToBf16.Source
