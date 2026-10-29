import bench.tritonbench_g.parallel_attention.Proofs.FPRange
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

def firstBody (BTL BTS BK BV : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := (forwardSource "q" "k" "v" "o" "z" 1 1 1 1 1 1 1 1 1 1 1 1 BTL BTS BK BV).surfaceBody
  match code[13]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
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
    (s t : State α) (start : Nat) (acc : ForwardAccumulator α c.BTL c.BV) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  query : t.regs .real [c.BTL,c.BK] "b_q" = some (forwardQuery M c s)
  output : t.regs .real [c.BTL,c.BV] "b_o" = some acc.output
  normalizer : t.regs .real [c.BTL] "b_z" = some acc.normalizer
  key : t.regs .blockPtr [c.BK,c.BTS] "p_k" = some (fun _ => keyPointer c s start)
  value : t.regs .blockPtr [c.BTS,c.BV] "p_v" = some (fun _ => valuePointer c s start)
  chunk : t.regs .nat [] "i_c" = some (fun _ => s.pids 1)
  head : t.regs .nat [] "i_bh" = some (fun _ => s.pids 2)
  keyBlock : t.regs .nat [] "i_k" = some (fun _ => s.pids 0 / ((c.V+c.BV-1)/c.BV))
  valueBlock : t.regs .nat [] "i_v" = some (fun _ => s.pids 0 % ((c.V+c.BV-1)/c.BV))

theorem first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start : Nat) (acc : ForwardAccumulator α c.BTL c.BV)
    (h : ForwardReady M c s t start acc) :
    ∃ final, run (withDot M D) (firstBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      ForwardReady M c s final (start+c.BTS) (forwardUpdate M D c s Bool.false start acc) := by
  simp [firstBody, forwardSource, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop,
    Region.cast, withDot, h.query, h.output, h.normalizer, h.key, h.value,
    h.mem, h.undef, h.pids, keyPointer, valuePointer, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    forwardUpdate, matrixRead]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query, forwardUpdate,
    keyPointer, valuePointer, matrixRead, withDot, bop, ofFloat, toFloat,
    h.chunk, h.head, h.keyBlock, h.valueBlock]
  all_goals rfl


theorem forwardReady_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : ForwardAccumulator α c.BTL c.BV)
    (h : ForwardReady M c s t start acc) (n : Nat) :
    ForwardReady M c s (t.setReg "_i" .nat [] (fun _ => n)) start acc := by
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query,
    h.output, h.normalizer, h.key, h.value, h.chunk, h.head, h.keyBlock, h.valueBlock]

theorem first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start stop : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : ForwardReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "_i" start stop c.BTS
        (firstBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      ∃ finish, ForwardReady M c s final finish
        (foldRange (forwardUpdate M D c s Bool.false) start stop c.BTS acc) := by
  apply range_values (withDot M D) "_i" (firstBody c.BTL c.BTS c.BK c.BV)
    (forwardUpdate M D c s Bool.false) (fun n a t => ForwardReady M c s t n a)
    (fun a t => ∃ n, ForwardReady M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact first_body_run M D c s _ n a (forwardReady_set_index M c s u n a hu n)
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h

def secondBody (BTL BTS BK BV : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := (forwardSource "q" "k" "v" "o" "z" 1 1 1 1 1 1 1 1 1 1 1 1 BTL BTS BK BV).surfaceBody
  match code[19]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

structure DiagonalReady {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : ForwardAccumulator α c.BTL c.BV)
    : Prop extends ForwardReady M c s t start acc where
  lower : queryChunk c s ≤ start
  queryIndex : t.regs .nat [c.BTL] "o_q" = some (fun i => i.1.val)
  keyIndex : t.regs .nat [c.BTS] "o_k" = some (fun i => start-queryChunk c s+i.1.val)

theorem second_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start : Nat) (acc : ForwardAccumulator α c.BTL c.BV)
    (h : DiagonalReady M c s t start acc) :
    ∃ final, run (withDot M D) (secondBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      DiagonalReady M c s final (start+c.BTS) (forwardUpdate M D c s Bool.true start acc) := by
  simp [secondBody, forwardSource, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop,
    Region.cast, withDot, h.query, h.output, h.normalizer, h.key, h.value,
    h.queryIndex, h.keyIndex, h.mem, h.undef, h.pids,
    keyPointer, valuePointer, TileShape.indexToList, numericLt, numericLe, natLe, natLt,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    forwardUpdate, matrixRead, ofFloat, toFloat]
  constructor
  · constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query, forwardUpdate,
      keyPointer, valuePointer, matrixRead, withDot, bop, ofFloat, toFloat,
    h.chunk, h.head, h.keyBlock, h.valueBlock]
    all_goals rfl
  · have := h.lower
    omega
  · simp [State.setReg, h.queryIndex]
  · simp [State.setReg, h.keyIndex, bop, numeric]
    funext i
    have := h.lower
    change start - queryChunk c s + i.1.val + c.BTS = start + c.BTS - queryChunk c s + i.1.val
    omega


theorem diagonalReady_set_index {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (start : Nat) (acc : ForwardAccumulator α c.BTL c.BV)
    (h : DiagonalReady M c s t start acc) (n : Nat) :
    DiagonalReady M c s (t.setReg "_i" .nat [] (fun _ => n)) start acc where
  toForwardReady := forwardReady_set_index M c s t start acc h.toForwardReady n
  lower := h.lower
  queryIndex := by simpa [State.setReg] using h.queryIndex
  keyIndex := by simpa [State.setReg] using h.keyIndex

theorem second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (start stop : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : DiagonalReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "_i" start stop c.BTS
        (secondBody c.BTL c.BTS c.BK c.BV) t = some final ∧
      ∃ finish, ForwardReady M c s final finish
        (foldRange (forwardUpdate M D c s Bool.true) start stop c.BTS acc) := by
  apply range_values (withDot M D) "_i" (secondBody c.BTL c.BTS c.BK c.BV)
    (forwardUpdate M D c s Bool.true) (fun n a t => DiagonalReady M c s t n a)
    (fun a t => ∃ n, ForwardReady M c s t n a) start stop c.BTS acc t
  · intro n a u hu _
    exact second_body_run M D c s _ n a (diagonalReady_set_index M c s u n a hu n)
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
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
