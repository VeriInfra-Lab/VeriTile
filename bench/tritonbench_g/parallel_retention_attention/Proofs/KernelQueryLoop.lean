/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryGraph
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig foldRange range_values)
open FPStages (logDecay rowDecay blockDecay)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

def firstBody (c : BackwardConfig) : List ComputeStmt :=
  match (source c "dq").surfaceBody[8]? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []

def secondBody (c : BackwardConfig) : List ComputeStmt :=
  match (source c "dq").surfaceBody[14]? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []

def keyPointer (c : BackwardConfig) (start : Nat) : BlockPtr :=
  .mk c.k (c.i_bh*c.s_qk_h) [c.T,c.K] [c.BTS,c.BK] [c.s_qk_t,c.s_qk_d]
    [start,c.i_k*c.BK]

def valuePointer (c : BackwardConfig) (start : Nat) : BlockPtr :=
  .mk c.v (c.i_bh*c.s_vo_h) [c.V,c.T] [c.BV,c.BTS] [c.s_vo_d,c.s_vo_t]
    [c.i_v*c.BV,start]

/-- The original loop's live registers and immutable inputs. Accumulator
values come from the independent initial-memory recurrence. -/
structure Ready {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  outGrad : t.regs .real [c.BTL,c.BV] "b_do" = some (outputGradient M c s)
  queryGrad : t.regs .real [c.BTL,c.BK] "b_dq" = some acc
  decayLog : t.regs .real [] "b_b" = some (fun _ => logDecay M c.i_h)
  decayBlock : t.regs .real [] "d_b" = some (fun _ => blockDecay M (logDecay M c.i_h) c.BTS)
  decay : t.regs .real [c.BTS] "d_h" = some (fun i => rowDecay M (logDecay M c.i_h) (c.BTS-i.1.val))
  key : t.regs .blockPtr [c.BTS,c.BK] "p_k" = some (fun _ => keyPointer c start)
  value : t.regs .blockPtr [c.BV,c.BTS] "p_v" = some (fun _ => valuePointer c start)

theorem first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) :
    ∃ final, run (withDot M D) (firstBody c) t = some final ∧
      Ready M c s final (start+c.BTS) (update M D c s Bool.false start acc) := by
  simp [firstBody, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, boundaryZero, Option.bind, ComputeDType.eraseDType, numeric, bop,
    Region.cast, withDot, h.outGrad, h.queryGrad, h.decayLog, h.decayBlock, h.decay, h.key, h.value,
    h.mem, h.undef, h.pids, keyPointer, valuePointer, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    update, matrixRead, rowDecay, blockDecay]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.outGrad, h.decayLog,
    h.decayBlock, h.decay, update, keyPointer, valuePointer, matrixRead,
    rowDecay, blockDecay, withDot, bop, ofFloat, toFloat]
  all_goals rfl

theorem ready_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) (n : Nat) :
    Ready M c s (t.setReg "_i" .nat [] (fun _ => n)) start acc := by
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.outGrad,
    h.queryGrad, h.decayLog, h.decayBlock, h.decay, h.key, h.value]

theorem first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : Ready M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "_i" start stop c.BTS (firstBody c) t = some final ∧
      ∃ finish, Ready M c s final finish
        (foldRange (update M D c s Bool.false) start stop c.BTS acc) := by
  apply range_values (withDot M D) "_i" (firstBody c)
    (update M D c s Bool.false) (fun n a t => Ready M c s t n a)
    (fun a t => ∃ n, Ready M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact first_body_run M D c s _ n a (ready_set_index M c s u n a hu n)
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h

structure DiagonalReady {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    : Prop extends Ready M c s t start acc where
  lower : c.i_c*c.BTL ≤ start
  queryIndex : t.regs .nat [c.BTL] "o_q" = some (fun i => i.1.val)
  keyIndex : t.regs .nat [c.BTS] "o_k" = some (fun i => start-c.i_c*c.BTL+i.1.val)

theorem second_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : DiagonalReady M c s t start acc) :
    ∃ final, run (withDot M D) (secondBody c) t = some final ∧
      DiagonalReady M c s final (start+c.BTS) (update M D c s Bool.true start acc) := by
  simp [secondBody, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, boundaryZero, Option.bind, ComputeDType.eraseDType, numeric, bop,
    Region.cast, withDot, h.outGrad, h.queryGrad, h.decayLog, h.decayBlock, h.decay, h.key, h.value,
    h.queryIndex, h.keyIndex, h.mem, h.undef, h.pids,
    keyPointer, valuePointer, TileShape.indexToList, numericLt, numericLe, natLe, natLt,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    update, matrixRead, rowDecay, blockDecay, ofFloat, toFloat]
  constructor
  · constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.outGrad, h.decayLog,
      h.decayBlock, h.decay, update, keyPointer, valuePointer, matrixRead,
      rowDecay, blockDecay, withDot, bop, ofFloat, toFloat]
    all_goals rfl
  · have := h.lower
    omega
  · simp [State.setReg, h.queryIndex]
  · simp [State.setReg, h.keyIndex, bop, numeric]
    funext i
    have := h.lower
    change start - c.i_c*c.BTL + i.1.val + c.BTS = start + c.BTS - c.i_c*c.BTL + i.1.val
    omega

theorem diagonal_ready_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : DiagonalReady M c s t start acc) (n : Nat) :
    DiagonalReady M c s (t.setReg "_i" .nat [] (fun _ => n)) start acc where
  toReady := ready_set_index M c s t start acc h.toReady n
  lower := h.lower
  queryIndex := by simpa [State.setReg] using h.queryIndex
  keyIndex := by simpa [State.setReg] using h.keyIndex

theorem second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : DiagonalReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "_i" start stop c.BTS (secondBody c) t = some final ∧
      ∃ finish, Ready M c s final finish
        (foldRange (update M D c s Bool.true) start stop c.BTS acc) := by
  apply range_values (withDot M D) "_i" (secondBody c)
    (update M D c s Bool.true) (fun n a t => DiagonalReady M c s t n a)
    (fun a t => ∃ n, Ready M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact second_body_run M D c s _ n a (diagonal_ready_set_index M c s u n a hu n)
  · intro n a u hu
    exact ⟨n,hu.toReady⟩
  · exact h

#axiomsClean first_body_run
#axiomsClean second_body_run
#axiomsClean first_range_run
#axiomsClean second_range_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
