import bench.tritonbench_g.parallel_retention_attention.Proofs.FPBackwardGraph
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig BackwardAccumulator matrixRead matrixTranspose backwardKey backwardValue descendingCount descendingStart foldRange backwardOutputMemory range_values)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def backwardFirstBody (c : BackwardConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (backwardKernel c).surfaceBody
  match code[11]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def backwardSecondBody (c : BackwardConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (backwardKernel c).surfaceBody
  match code[17]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

structure BackwardReady {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  key : t.regs .real [c.BTL,c.BK] "b_k" = some (backwardKey M c s)
  value : t.regs .real [c.BTL,c.BV] "b_v" = some (backwardValue M c s)
  keyGrad : t.regs .real [c.BTL,c.BK] "b_dk" = some acc.key
  valueGrad : t.regs .real [c.BTL,c.BV] "b_dv" = some acc.value
  decayLog : t.regs .real [] "b_b" = some (fun _ => logDecay M c.i_h)
  decayBlock : t.regs .real [] "d_b" = some (fun _ => blockDecay M (logDecay M c.i_h) c.BTS)
  decayHead : t.regs .real [c.BTL] "d_h" = some (fun i => rowDecay M (logDecay M c.i_h) (c.BTL-i.1.val))
  decayQuery : t.regs .real [c.BTS] "d_q" = some (fun i => rowDecay M (logDecay M c.i_h) i.1.val)
  keyDecayed : t.regs .real [c.BTL,c.BK] "b_kd" = some (decayedKey M c s)

theorem backward_first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (j : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hj : t.regs .nat [] "j" = some (fun _ => j)) :
    ∃ final, run (withDot M D) (backwardFirstBody c) t = some final ∧
      BackwardReady M c s final (backwardUpdate M D c s Bool.false (descendingStart c j) acc) := by
  simp [ backwardFirstBody, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed, hj, h.mem, h.pids, h.undef,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt]
  all_goals constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.key, h.value, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,
    backwardUpdate, descendingStart, matrixRead, matrixTranspose, transposeValues, rowDecay, blockDecay, decayedKey,
    bop, numeric, ofFloat, toFloat]
  all_goals rfl

structure BackwardDiagonalReady {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    : Prop extends BackwardReady M c s t acc where
  lower : c.i_c*c.BTL ≤ start
  queryIndex : t.regs .nat [c.BTS] "o_q" = some (fun i => start-c.i_c*c.BTL+i.1.val)
  keyIndex : t.regs .nat [c.BTL] "o_k" = some (fun i => i.1.val)

theorem backward_second_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc)
    (hi : t.regs .nat [] "i" = some (fun _ => start)) :
    ∃ final, run (withDot M D) (backwardSecondBody c) t = some final ∧
      BackwardDiagonalReady M c s final (start+c.BTS) (backwardUpdate M D c s Bool.true start acc) := by
  simp [ backwardSecondBody, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed, hi, h.mem, h.pids, h.undef, h.queryIndex, h.keyIndex,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt, natLe, numericLe]
  all_goals constructor
  all_goals first
    | (constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.key, h.value, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,
        backwardUpdate, matrixRead, matrixTranspose, transposeValues, rowDecay, blockDecay, decayedKey,
        bop, numeric, ofFloat, toFloat]; all_goals rfl)
    | (have := h.lower; omega)
    | (simp [State.setReg, h.queryIndex, h.keyIndex]; funext i;
        change start-c.i_c*c.BTL+i.1.val+c.BTS = start+c.BTS-c.i_c*c.BTL+i.1.val;
        have := h.lower; omega)
    | (simp [State.setReg, h.keyIndex])

theorem backwardReady_j {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (j : Nat) :
    BackwardReady M c s (t.setReg "j" .nat [] (fun _ => j)) acc := by
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed]

theorem backward_first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardReady M c s t acc) :
    ∃ final, FP.Structural.range (withDot M D) "j" start stop 1 (backwardFirstBody c) t = some final ∧
      BackwardReady M c s final
        (foldRange (fun j => backwardUpdate M D c s Bool.false (descendingStart c j)) start stop 1 acc) := by
  apply range_values (withDot M D) "j" (backwardFirstBody c)
    (fun j => backwardUpdate M D c s Bool.false (descendingStart c j))
    (fun _ a t => BackwardReady M c s t a) (fun a t => BackwardReady M c s t a)
    start stop 1 acc t
  · intro n a u hu _
    exact backward_first_body_run M D c s _ n a (backwardReady_j M c s u a hu n) (by simp [State.setReg])
  · intro n a u hu
    exact hu
  · exact h

theorem backwardDiagonalReady_i {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc) (i : Nat) :
    BackwardDiagonalReady M c s (t.setReg "i" .nat [] (fun _ => i)) start acc := by
  constructor
  · constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed]
  · exact h.lower
  · simpa [State.setReg] using h.queryIndex
  · simpa [State.setReg] using h.keyIndex

theorem backward_second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardDiagonalReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "i" start stop c.BTS (backwardSecondBody c) t = some final ∧
      BackwardReady M c s final (foldRange (backwardUpdate M D c s Bool.true) start stop c.BTS acc) := by
  apply range_values (withDot M D) "i" (backwardSecondBody c)
    (backwardUpdate M D c s Bool.true) (fun n a t => BackwardDiagonalReady M c s t n a)
    (fun a t => BackwardReady M c s t a) start stop c.BTS acc t
  · intro n a u hu _
    exact backward_second_body_run M D c s _ n a (backwardDiagonalReady_i M c s u n a hu n) (by simp [State.setReg])
  · intro n a u hu
    exact hu.toBackwardReady
  · exact h

#axiomsClean backward_first_body_run
#axiomsClean backward_second_body_run
#axiomsClean backward_first_range_run
#axiomsClean backward_second_range_run
end
end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
