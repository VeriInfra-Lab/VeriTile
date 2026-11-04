/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryGraph
import bench.tritonbench_g.parallel_retention_attention.Proofs.SignedRange
import bench.tritonbench_g.parallel_retention_attention.Proofs.Index32

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig foldRange)
open FPStages (logDecay rowDecay blockDecay)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

def firstBody (c : BackwardConfig) : List ComputeStmt :=
  match (source c "dq").surfaceBody[36]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

def secondBody (c : BackwardConfig) : List ComputeStmt :=
  match (source c "dq").surfaceBody[42]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
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

  argument0 : t.regs .int [] "__veritile_argument_18379874528259158415_0"=some (fun _=>(c.i_bh : Int))
  argument1 : t.regs .int [] "__veritile_argument_18379874528259158415_1"=some (fun _=>(c.i_c : Int))
  argument2 : t.regs .int [] "__veritile_argument_18379874528259158415_2"=some (fun _=>(c.i_k : Int))
  argument3 : t.regs .int [] "__veritile_argument_18379874528259158415_3"=some (fun _=>(c.i_v : Int))
  argument4 : t.regs .int [] "__veritile_argument_18379874528259158415_4"=some (fun _=>(c.i_h : Int))
  argument5 : t.regs .int [] "__veritile_argument_18379874528259158415_5"=some (fun _=>(c.s_qk_h : Int))
  argument6 : t.regs .int [] "__veritile_argument_18379874528259158415_6"=some (fun _=>(c.s_qk_t : Int))
  argument7 : t.regs .int [] "__veritile_argument_18379874528259158415_7"=some (fun _=>(c.s_qk_d : Int))
  argument8 : t.regs .int [] "__veritile_argument_18379874528259158415_8"=some (fun _=>(c.s_vo_h : Int))
  argument9 : t.regs .int [] "__veritile_argument_18379874528259158415_9"=some (fun _=>(c.s_vo_t : Int))
  argument10 : t.regs .int [] "__veritile_argument_18379874528259158415_10"=some (fun _=>(c.s_vo_d : Int))
  argument11 : t.regs .int [] "__veritile_argument_18379874528259158415_11"=some (fun _=>(c.B : Int))
  argument12 : t.regs .int [] "__veritile_argument_18379874528259158415_12"=some (fun _=>(c.H : Int))
  argument13 : t.regs .int [] "__veritile_argument_18379874528259158415_13"=some (fun _=>(c.T : Int))

  outerHead : t.regs .int [] "i_bh"=s.regs .int [] "i_bh"
  outerChunk : t.regs .int [] "i_c"=s.regs .int [] "i_c"
  outerKey : t.regs .int [] "i_k"=s.regs .int [] "i_k"
  outerValue : t.regs .int [] "i_v"=s.regs .int [] "i_v"
  outerDecay : t.regs .int [] "i_h"=s.regs .int [] "i_h"

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
    rowDecay, blockDecay, withDot, bop, ofFloat, toFloat,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay]
  all_goals rfl

theorem ready_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) (n : Nat) :
    Ready M c s (t.setReg "_" .int [] (fun _ => (n : Int))) start acc := by
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.outGrad,
    h.queryGrad, h.decayLog, h.decayBlock, h.decay, h.key, h.value,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay]

theorem first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : Ready M c s t start acc) :
    ∃ final, FP.Structural.rangeInt (withDot M D) "_" (start : Int) (stop : Int) c.BTS (firstBody c) t = some final ∧
      ∃ finish, Ready M c s final finish
        (foldRange (update M D c s Bool.false) start stop c.BTS acc) := by
  apply rangeInt_values (withDot M D) "_" (firstBody c)
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
    (h : DiagonalReady M c s t start acc) (laneBound : c.BTL≤2147483648) :
    ∃ final, run (withDot M D) (secondBody c) t = some final ∧
      DiagonalReady M c s final (start+c.BTS) (update M D c s Bool.true start acc) := by
  have mask (row : Fin c.BTL) (column : Nat) :
      (if column≤row.val then
        M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
          (M.fromInt (some .fp32) (Index32.signed row.val-Index32.signed column))
          (logDecay M c.i_h)) else M.literal (some .fp32) .real 0)=
      if column≤row.val then
        M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
          (M.fromInt (some .fp32) ((row.val : Int)-column))
          (logDecay M c.i_h)) else M.literal (some .fp32) .real 0 :=
    Index32.masked row.val column (Nat.lt_of_lt_of_le row.isLt laneBound)
      (fun delta=>M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
        (M.fromInt (some .fp32) delta) (logDecay M c.i_h)))
      (M.literal (some .fp32) .real 0)
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
      rowDecay, blockDecay, withDot, bop, ofFloat, toFloat,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay]
    all_goals simp only [Index32.expansion (α := α),←Int.natCast_add,mask]
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
    DiagonalReady M c s (t.setReg "_" .int [] (fun _ => (n : Int))) start acc where
  toReady := ready_set_index M c s t start acc h.toReady n
  lower := h.lower
  queryIndex := by simpa [State.setReg] using h.queryIndex
  keyIndex := by simpa [State.setReg] using h.keyIndex

theorem second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : DiagonalReady M c s t start acc) (laneBound : c.BTL≤2147483648) :
    ∃ final, FP.Structural.rangeInt (withDot M D) "_" (start : Int) (stop : Int) c.BTS (secondBody c) t = some final ∧
      ∃ finish, Ready M c s final finish
        (foldRange (update M D c s Bool.true) start stop c.BTS acc) := by
  apply rangeInt_values (withDot M D) "_" (secondBody c)
    (update M D c s Bool.true) (fun n a t => DiagonalReady M c s t n a)
    (fun a t => ∃ n, Ready M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact second_body_run M D c s _ n a (diagonal_ready_set_index M c s u n a hu n) laneBound
  · intro n a u hu
    exact ⟨n,hu.toReady⟩
  · exact h

#axiomsClean first_body_run
#axiomsClean second_body_run
#axiomsClean first_range_run
#axiomsClean second_range_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
