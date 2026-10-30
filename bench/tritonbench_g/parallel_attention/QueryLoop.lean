import bench.tritonbench_g.parallel_attention.QueryGraph
import bench.tritonbench_g.parallel_attention.FPRange

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

@[simp] theorem matrix_index_eta (i : TileIndex [m,n]) : (⟨i.1,i.2.1,()⟩ : TileIndex [m,n])=i := by
  rcases i with ⟨r,e,⟨⟩⟩; rfl

theorem matrixRead_zero_row {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (base rows cols rowStride colStride colOffset m n : Nat) :
    matrixRead M s R base rows cols rowStride colStride 0 colOffset m n =
      fun i => M.fp32Load (if i.1.val<rows ∧ colOffset+i.2.1.val<cols then
        (s.mem R (base+i.1.val*rowStride+(colOffset+i.2.1.val)*colStride)).read .real
        else M.literal (some .fp32) .real 0) := by
  funext i
  simp only [matrixRead,Nat.zero_add]

def queryFirstBody (c : QueryConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (queryKernel c).surfaceBody
  match code[10]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def querySecondBody (c : QueryConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (queryKernel c).surfaceBody
  match code[16]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

def queryKeyPointer (c : QueryConfig) (start : Nat) : BlockPtr :=
  .mk c.k (c.i_bh*c.s_qk_h) [c.T,c.K] [c.BTS,c.BK] [c.s_qk_t,c.s_qk_d]
    [start,c.i_k*c.BK]
def queryValuePointer (c : QueryConfig) (start : Nat) : BlockPtr :=
  .mk c.v (c.i_bh*c.s_vo_h) [c.V,c.T] [c.BV,c.BTS] [c.s_vo_d,c.s_vo_t]
    [c.i_v*c.BV,start]

structure QueryReady {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  query : t.regs .real [c.BTL,c.BK] "b_q" = some (queryInput M c s)
  outGrad : t.regs .real [c.BTL,c.BV] "b_do" = some (queryOutGrad M c s)
  normGrad : t.regs .real [c.BTL] "b_dz" = some (queryNormGrad M c s)
  queryGrad : t.regs .real [c.BTL,c.BK] "b_dq" = some acc
  key : t.regs .blockPtr [c.BTS,c.BK] "p_k" = some (fun _ => queryKeyPointer c start)
  value : t.regs .blockPtr [c.BV,c.BTS] "p_v" = some (fun _ => queryValuePointer c start)

theorem query_first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryReady M c s t start acc) :
    ∃ final, run (withDot M D) (queryFirstBody c) t = some final ∧
      QueryReady M c s final (start+c.BTS) (queryUpdate M D c s Bool.false start acc) := by
  by_cases hv : c.i_v=0
  all_goals simp [hv, queryFirstBody, queryKernel, querySource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.query, h.outGrad, h.normGrad, h.queryGrad, h.key, h.value, h.mem, h.pids, h.undef,
    queryKeyPointer, queryValuePointer, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets]
  all_goals constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query,
    h.outGrad, h.normGrad, queryUpdate, queryKeyPointer, queryValuePointer,
    matrixRead, matrixTranspose, transposeValues, hv, withDot, bop, ofFloat, toFloat, Int.toNat_mul]
  all_goals try dsimp only [matrixRead,matrixTranspose,bop]
  all_goals try simp only [Nat.zero_add,matrix_index_eta]
  all_goals try simp only [matrixRead_zero_row]
  all_goals rfl

structure QueryDiagonalReady {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    : Prop extends QueryReady M c s t start acc where
  lower : c.i_c*c.BTL ≤ start
  queryIndex : t.regs .nat [c.BTL] "o_q" = some (fun i => i.1.val)
  keyIndex : t.regs .nat [c.BTS] "o_k" = some (fun i => start-c.i_c*c.BTL+i.1.val)

theorem query_second_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryDiagonalReady M c s t start acc) :
    ∃ final, run (withDot M D) (querySecondBody c) t = some final ∧
      QueryDiagonalReady M c s final (start+c.BTS) (queryUpdate M D c s Bool.true start acc) := by
  by_cases hv : c.i_v=0
  all_goals simp [hv, querySecondBody, queryKernel, querySource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.query, h.outGrad, h.normGrad, h.queryGrad, h.key, h.value, h.mem, h.pids, h.undef,
    h.queryIndex, h.keyIndex, queryKeyPointer, queryValuePointer, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, BlockPtr.advance_2d_offsets,
    ofFloat, toFloat, natLe, numericLe]
  all_goals constructor
  case pos.toQueryReady | neg.toQueryReady =>
    constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query,
      h.outGrad, h.normGrad, queryUpdate, queryKeyPointer, queryValuePointer,
      matrixRead, matrixTranspose, transposeValues, hv, withDot, bop, ofFloat, toFloat, Int.toNat_mul]
    all_goals try dsimp only [matrixRead,matrixTranspose,bop]
    all_goals try simp only [Nat.zero_add,matrix_index_eta]
    all_goals try simp only [matrixRead_zero_row]
    all_goals rfl
  case pos.lower | neg.lower => have := h.lower; omega
  case pos.queryIndex | neg.queryIndex => simp [State.setReg,h.queryIndex]
  case pos.keyIndex | neg.keyIndex =>
    simp [State.setReg,h.keyIndex,bop,numeric]
    funext i
    change start-c.i_c*c.BTL+i.1.val+c.BTS = start+c.BTS-c.i_c*c.BTL+i.1.val
    have := h.lower
    omega

theorem queryReady_index {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryReady M c s t start acc) (n : Nat) :
    QueryReady M c s (t.setReg "_i" .nat [] (fun _ => n)) start acc := by
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query, h.outGrad,
    h.normGrad, h.queryGrad, h.key, h.value]

theorem queryDiagonalReady_index {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryDiagonalReady M c s t start acc) (n : Nat) :
    QueryDiagonalReady M c s (t.setReg "_i" .nat [] (fun _ => n)) start acc where
  toQueryReady := queryReady_index M c s t start acc h.toQueryReady n
  lower := h.lower
  queryIndex := by simpa [State.setReg] using h.queryIndex
  keyIndex := by simpa [State.setReg] using h.keyIndex

theorem query_first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start stop : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "_i" start stop c.BTS (queryFirstBody c) t = some final ∧
      ∃ finish, QueryReady M c s final finish
        (foldRange (queryUpdate M D c s Bool.false) start stop c.BTS acc) := by
  apply range_values (withDot M D) "_i" (queryFirstBody c) (queryUpdate M D c s Bool.false)
    (fun n a t => QueryReady M c s t n a) (fun a t => ∃ n, QueryReady M c s t n a)
    start stop c.BTS acc t
  · intro n a u hu _
    exact query_first_body_run M D c s _ n a (queryReady_index M c s u n a hu n)
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h

theorem query_second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start stop : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryDiagonalReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "_i" start stop c.BTS (querySecondBody c) t = some final ∧
      ∃ finish, QueryReady M c s final finish
        (foldRange (queryUpdate M D c s Bool.true) start stop c.BTS acc) := by
  apply range_values (withDot M D) "_i" (querySecondBody c) (queryUpdate M D c s Bool.true)
    (fun n a t => QueryDiagonalReady M c s t n a) (fun a t => ∃ n, QueryReady M c s t n a)
    start stop c.BTS acc t
  · intro n a u hu _
    exact query_second_body_run M D c s _ n a (queryDiagonalReady_index M c s u n a hu n)
  · intro n a u hu
    exact ⟨n,hu.toQueryReady⟩
  · exact h

#axiomsClean query_first_body_run
#axiomsClean query_second_body_run
#axiomsClean query_first_range_run
#axiomsClean query_second_range_run
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
