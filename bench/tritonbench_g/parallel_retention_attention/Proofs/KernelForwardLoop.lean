/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardGraph
import bench.tritonbench_g.parallel_retention_attention.Proofs.SignedRange
import bench.tritonbench_g.parallel_retention_attention.Proofs.Index32
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot ForwardConfig queryChunk keyChunk valueChunk foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

def firstBody (BTL BTS BK BV : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := (forwardSource "q" "k" "v" "o" 1 1 1 1 1 1 1 1 1 1 1 1 BTL BTS BK BV).surfaceBody
  match code[32]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

def keyPointer {α : Type} (c : ForwardConfig) (s : State α) (start : Nat) : BlockPtr :=
  .mk c.k (s.pids 2 * c.s_qk_h) [c.K,c.T] [c.BK,c.BTS] [c.s_qk_d,c.s_qk_t]
    [keyChunk c s,start]
def valuePointer {α : Type} (c : ForwardConfig) (s : State α) (start : Nat) : BlockPtr :=
  .mk c.v (s.pids 2 * c.s_vo_h) [c.T,c.V] [c.BTS,c.BV] [c.s_vo_t,c.s_vo_d]
    [start,valueChunk c s]

/-- Only live registers and immutable memory inputs constrain a loop state;
temporary registers remain unrestricted. The carried values are independent
arrays from the numerical recurrence. -/
structure ForwardReady {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  query : t.regs .real [c.BTL,c.BK] "b_q" = some (forwardQuery M c s)
  output : t.regs .real [c.BTL,c.BV] "b_o" = some acc
  decayLog : t.regs .real [] "b_b" = some (fun _ => logDecay M (s.pids 2 % c.H))
  decay : t.regs .real [c.BTS] "d_h" = some (fun i => rowDecay M (logDecay M (s.pids 2 % c.H)) (c.BTS-i.1.val))
  key : t.regs .blockPtr [c.BK,c.BTS] "p_k" = some (fun _ => keyPointer c s start)
  value : t.regs .blockPtr [c.BTS,c.BV] "p_v" = some (fun _ => valuePointer c s start)
  chunk : t.regs .int [] "i_c" = some (fun _ => ((s.pids 1) : Int))
  head : t.regs .int [] "i_bh" = some (fun _ => ((s.pids 2) : Int))
  keyBlock : t.regs .int [] "i_k" = some (fun _ => ((s.pids 0 / ((c.V+c.BV-1)/c.BV) : Nat) : Int))
  valueBlock : t.regs .int [] "i_v" = some (fun _ => ((s.pids 0 % ((c.V+c.BV-1)/c.BV) : Nat) : Int))

  argument0 : t.regs .int [] "__veritile_argument_12546945468679417523_0" = some (fun _ => (c.s_qk_h : Int))
  argument1 : t.regs .int [] "__veritile_argument_12546945468679417523_1" = some (fun _ => (c.s_qk_t : Int))
  argument2 : t.regs .int [] "__veritile_argument_12546945468679417523_2" = some (fun _ => (c.s_qk_d : Int))
  argument3 : t.regs .int [] "__veritile_argument_12546945468679417523_3" = some (fun _ => (c.s_vo_h : Int))
  argument4 : t.regs .int [] "__veritile_argument_12546945468679417523_4" = some (fun _ => (c.s_vo_t : Int))
  argument5 : t.regs .int [] "__veritile_argument_12546945468679417523_5" = some (fun _ => (c.s_vo_d : Int))
  argument6 : t.regs .int [] "__veritile_argument_12546945468679417523_6" = some (fun _ => (c.B : Int))
  argument7 : t.regs .int [] "__veritile_argument_12546945468679417523_7" = some (fun _ => (c.H : Int))
  argument8 : t.regs .int [] "__veritile_argument_12546945468679417523_8" = some (fun _ => (c.T : Int))

theorem first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : ForwardReady M c s t start acc) :
    ∃ final, run (withDot M D) (firstBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      ForwardReady M c s final (start+c.BTS) (forwardUpdate M D c s Bool.false start acc) := by
  simp [firstBody, forwardSource, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, boundaryZero, Option.bind, ComputeDType.eraseDType, numeric, bop,
    Region.cast, withDot, h.query, h.output, h.decayLog, h.decay, h.key, h.value,
    h.mem, h.undef, h.pids, keyPointer, valuePointer, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    forwardUpdate, matrixRead, rowDecay, blockDecay]
  constructor <;> simp [h.mem, h.pids, h.undef, h.query, h.decayLog, h.decay, forwardUpdate,
    keyPointer, valuePointer, matrixRead, rowDecay, blockDecay, withDot, bop, ofFloat, toFloat,
    h.chunk, h.head, h.keyBlock, h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8]
  all_goals rfl


theorem forwardReady_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : ForwardReady M c s t start acc) (n : Nat) :
    ForwardReady M c s (t.setReg "_" .int [] (fun _ => (n : Int))) start acc := by
  constructor <;> simp [h.mem, h.pids, h.undef, h.query,
    h.output, h.decayLog, h.decay, h.key, h.value, h.chunk, h.head, h.keyBlock, h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8]

theorem first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BV] → α) (h : ForwardReady M c s t start acc) :
    ∃ final, FP.Structural.rangeInt (withDot M D) "_" (start : Int) (stop : Int) c.BTS
        (firstBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      ∃ finish, ForwardReady M c s final finish
        (foldRange (forwardUpdate M D c s Bool.false) start stop c.BTS acc) := by
  apply rangeInt_values (withDot M D) "_" (firstBody c.BTL c.BTS c.BK c.BV)
    (forwardUpdate M D c s Bool.false) (fun n a t => ForwardReady M c s t n a)
    (fun a t => ∃ n, ForwardReady M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact first_body_run M D c s _ n a (forwardReady_set_index M c s u n a hu n)
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h

def secondBody (BTL BTS BK BV : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := (forwardSource "q" "k" "v" "o" 1 1 1 1 1 1 1 1 1 1 1 1 BTL BTS BK BV).surfaceBody
  match code[40]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

structure DiagonalReady {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    : Prop extends ForwardReady M c s t start acc where
  lower : queryChunk c s ≤ start
  queryIndex : t.regs .nat [c.BTL] "o_q" = some (fun i => i.1.val)
  keyIndex : t.regs .nat [c.BTS] "o_k" = some (fun i => start-queryChunk c s+i.1.val)

theorem second_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : DiagonalReady M c s t start acc) (laneBound : c.BTL≤2147483648) :
    ∃ final, run (withDot M D) (secondBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      DiagonalReady M c s final (start+c.BTS) (forwardUpdate M D c s Bool.true start acc) := by
  have mask (row : Fin c.BTL) (column : Nat) :
      (if column≤row.val then
        M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
          (M.fromInt (some .fp32) (Index32.signed row.val-Index32.signed column))
          (logDecay M (s.pids 2%c.H))) else M.literal (some .fp32) .real 0)=
      if column≤row.val then
        M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
          (M.fromInt (some .fp32) ((row.val : Int)-column))
          (logDecay M (s.pids 2%c.H))) else M.literal (some .fp32) .real 0 :=
    Index32.masked row.val column (Nat.lt_of_lt_of_le row.isLt laneBound)
      (fun delta=>M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
        (M.fromInt (some .fp32) delta) (logDecay M (s.pids 2%c.H))))
      (M.literal (some .fp32) .real 0)
  simp [secondBody, forwardSource, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, boundaryZero, Option.bind, ComputeDType.eraseDType, numeric, bop,
    Region.cast, withDot, h.query, h.output, h.decayLog, h.decay, h.key, h.value,
    h.queryIndex, h.keyIndex, h.mem, h.undef, h.pids,
    keyPointer, valuePointer, TileShape.indexToList, numericLt, numericLe, natLe, natLt,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    forwardUpdate, matrixRead, rowDecay, blockDecay, ofFloat, toFloat]
  constructor
  · constructor <;> simp [State.setReg,h.mem, h.pids, h.undef, h.query, h.decayLog, h.decay, forwardUpdate,
      keyPointer, valuePointer, matrixRead, rowDecay, blockDecay, withDot, bop, ofFloat, toFloat,
    h.chunk, h.head, h.keyBlock, h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8]
    all_goals simp only [Index32.expansion (α := α),←Int.natCast_add,mask]
    all_goals rfl
  · have := h.lower
    omega
  · simp [h.queryIndex]
  · simp [h.keyIndex, bop, numeric]
    funext i
    have := h.lower
    change start - queryChunk c s + i.1.val + c.BTS = start + c.BTS - queryChunk c s + i.1.val
    omega


theorem diagonalReady_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : DiagonalReady M c s t start acc) (n : Nat) :
    DiagonalReady M c s (t.setReg "_" .int [] (fun _ => (n : Int))) start acc where
  toForwardReady := forwardReady_set_index M c s t start acc h.toForwardReady n
  lower := h.lower
  queryIndex := by simpa [State.setReg] using h.queryIndex
  keyIndex := by simpa [State.setReg] using h.keyIndex

theorem second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BV] → α) (h : DiagonalReady M c s t start acc) (laneBound : c.BTL≤2147483648) :
    ∃ final, FP.Structural.rangeInt (withDot M D) "_" (start : Int) (stop : Int) c.BTS
        (secondBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      ∃ finish, ForwardReady M c s final finish
        (foldRange (forwardUpdate M D c s Bool.true) start stop c.BTS acc) := by
  apply rangeInt_values (withDot M D) "_" (secondBody c.BTL c.BTS c.BK c.BV)
    (forwardUpdate M D c s Bool.true) (fun n a t => DiagonalReady M c s t n a)
    (fun a t => ∃ n, ForwardReady M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact second_body_run M D c s _ n a (diagonalReady_set_index M c s u n a hu n) laneBound
  · intro n a u hu
    exact ⟨n,hu.toForwardReady⟩
  · exact h
#axiomsClean diagonalReady_set_index
#axiomsClean second_range_run

#axiomsClean forwardReady_set_index
#axiomsClean first_range_run
#axiomsClean second_body_run

#axiomsClean first_body_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
