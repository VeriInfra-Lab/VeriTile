import bench.tritonbench_g.parallel_attention.Proofs.CallerQueryWhole

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def backwardFirstBody (c : QueryConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel c).surfaceBody
  match code[36]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []
def backwardSecondBody (c : QueryConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel c).surfaceBody
  match code[42]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

structure BackwardReady {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  key : t.regs .real [c.BTL,c.BK] "b_k" = some (backwardKey M c.toBackwardConfig s)
  value : t.regs .real [c.BTL,c.BV] "b_v" = some (backwardValue M c.toBackwardConfig s)
  keyGrad : t.regs .real [c.BTL,c.BK] "b_dk" = some acc.key
  valueGrad : t.regs .real [c.BTL,c.BV] "b_dv" = some acc.value
  head : t.regs .nat [] "i_bh"=some (fun _=>c.i_bh)
  chunk : t.regs .nat [] "i_c"=some (fun _=>c.i_c)
  keyBlock : t.regs .nat [] "i_k"=some (fun _=>c.i_k)
  valueBlock : t.regs .nat [] "i_v"=some (fun _=>c.i_v)

theorem signed_lane_lt (start lane bound : Nat) :
    (start : Int)+(lane : Int)<(bound : Int) ↔ start+lane<bound := by
  rw [←Nat.cast_add,Int.ofNat_lt]

theorem signed_lane_address (head width start lane : Nat) :
    Int.toNat ((head : Int)*(width : Int)+(start : Int)+(lane : Int))=head*width+start+lane := by
  rw [←Nat.cast_mul,←Nat.cast_add,←Nat.cast_add,Int.toNat_natCast]

theorem backward_first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hj : t.regs .int [] "__rev_i" = some (fun _ => -(start : Int))) :
    ∃ final, run (withDot M D) (backwardFirstBody c) t = some final ∧
      BackwardReady M c s final (backwardUpdate M D c.toBackwardConfig s Bool.false start acc) := by
  by_cases hv : c.i_v = 0
  all_goals simp [hv, backwardFirstBody, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.key,h.value,h.head,h.chunk,h.keyBlock,h.valueBlock, h.keyGrad, h.valueGrad, hj, h.mem, h.pids, h.undef,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt, Int.toNat_add, Int.toNat_mul]
  all_goals constructor <;> simp [h.head,h.chunk,h.keyBlock,h.valueBlock,h.mem, h.pids, h.undef, h.key,h.value,h.head,h.chunk,h.keyBlock,h.valueBlock,
    backwardUpdate, matrixRead, rawMatrixRead, matrixTranspose, transposeValues, hv,
    bop, numeric, ofFloat, toFloat]
  all_goals (first | rfl | (simp [bop, matrixRead, rawMatrixRead, matrixTranspose, signed_lane_lt, signed_lane_address]; rfl))

structure BackwardDiagonalReady {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    : Prop extends BackwardReady M c s t acc where
  lower : c.i_c*c.BTL ≤ start
  queryIndex : t.regs .nat [c.BTS] "o_q" = some (fun i => start-c.i_c*c.BTL+i.1.val)
  keyIndex : t.regs .nat [c.BTL] "o_k" = some (fun i => i.1.val)

theorem backward_second_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc)
    (hi : t.regs .nat [] "i" = some (fun _ => start)) :
    ∃ final, run (withDot M D) (backwardSecondBody c) t = some final ∧
      BackwardDiagonalReady M c s final (start+c.BTS) (backwardUpdate M D c.toBackwardConfig s Bool.true start acc) := by
  by_cases hv : c.i_v = 0
  all_goals simp [hv, backwardSecondBody, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.key,h.value,h.head,h.chunk,h.keyBlock,h.valueBlock, h.keyGrad, h.valueGrad, hi, h.mem, h.pids, h.undef, h.queryIndex, h.keyIndex,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt, natLe, numericLe]
  all_goals constructor
  all_goals first
    | (constructor <;> simp [ h.mem, h.pids, h.undef, h.key,h.value,h.head,h.chunk,h.keyBlock,h.valueBlock,
        backwardUpdate, matrixRead, rawMatrixRead, matrixTranspose, transposeValues, hv,
        bop, numeric, ofFloat, toFloat]; all_goals (first | rfl | (simp [bop, matrixRead, rawMatrixRead, matrixTranspose]; rfl)))
    | (have := h.lower; omega)
    | (simp [ h.queryIndex, h.keyIndex]; funext i;
        change start-c.i_c*c.BTL+i.1.val+c.BTS = start+c.BTS-c.i_c*c.BTL+i.1.val;
        have := h.lower; omega)
    | (simp [ h.keyIndex])

theorem backwardReady_j {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (index : Int) :
    BackwardReady M c s (t.setReg "__rev_i" .int [] (fun _ => index)) acc := by
  constructor <;> simp [h.head,h.chunk,h.keyBlock,h.valueBlock,h.mem, h.pids, h.undef, h.key,h.value,h.head,h.chunk,h.keyBlock,h.valueBlock, h.keyGrad, h.valueGrad]

theorem backward_first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start : Nat) (bound : start≤descendingCount c.toBackwardConfig)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardReady M c s t acc) :
    ∃ final, rangeInt (withDot M D) "__rev_i"
        (DescendingRange.lower c.toBackwardConfig+(start : Int)*(c.BTS : Int)) (DescendingRange.upper c.toBackwardConfig) c.BTS
        (backwardFirstBody c) t = some final ∧
      BackwardReady M c s final
        (foldRange (fun j => backwardUpdate M D c.toBackwardConfig s Bool.false (descendingStart c.toBackwardConfig j))
          start (descendingCount c.toBackwardConfig) 1 acc) := by
  by_cases zero : c.BTS=0
  · have count : descendingCount c.toBackwardConfig=0:=by simp [descendingCount,zero]
    have first : start=0:=by omega
    subst start
    exact ⟨t,by simp [rangeInt,zero],by simpa [foldRange,count] using h⟩
  have positive : 0<c.BTS:=by omega
  by_cases live : start<descendingCount c.toBackwardConfig
  · have current : DescendingRange.lower c.toBackwardConfig+(start : Int)*(c.BTS : Int)=
        -(descendingStart c.toBackwardConfig start : Int) := by
      have equal:=congrArg Neg.neg (DescendingRange.source_index c.toBackwardConfig positive start live)
      simpa only [neg_neg] using equal
    obtain ⟨middle,ran,maintained⟩:=backward_first_body_run M D c s _ (descendingStart c.toBackwardConfig start) acc
      (backwardReady_j M c s t acc h (-(descendingStart c.toBackwardConfig start : Int))) (by simp [State.setReg])
    obtain ⟨final,finished,invariant⟩:=backward_first_range_run M D c s middle (start+1) (by omega) _ maintained
    refine ⟨final,?_,?_⟩
    · rw [rangeInt]
      simp only [zero,↓reduceIte,(DescendingRange.source_active c.toBackwardConfig positive start).mpr live]
      rw [current,ran]
      have next : DescendingRange.lower c.toBackwardConfig+(start : Int)*(c.BTS : Int)+(c.BTS : Int)=
          DescendingRange.lower c.toBackwardConfig+((start+1 : Nat) : Int)*(c.BTS : Int) := by
        simp [Nat.cast_add,add_mul,add_assoc]
      rw [←current,next]
      exact finished
    · rw [foldRange]
      simpa only [Nat.one_ne_zero,live,↓reduceIte] using invariant
  · refine ⟨t,?_,?_⟩
    · rw [rangeInt]
      simp [zero,DescendingRange.source_active c.toBackwardConfig positive start,live]
    · simpa [foldRange,live] using h
termination_by descendingCount c.toBackwardConfig-start
decreasing_by omega

theorem backwardDiagonalReady_i {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc) (i : Nat) :
    BackwardDiagonalReady M c s (t.setReg "i" .nat [] (fun _ => i)) start acc := by
  constructor
  · constructor <;> simp [h.head,h.chunk,h.keyBlock,h.valueBlock,h.mem, h.pids, h.undef, h.key,h.value,h.head,h.chunk,h.keyBlock,h.valueBlock, h.keyGrad, h.valueGrad]
  · exact h.lower
  · simpa [State.setReg] using h.queryIndex
  · simpa [State.setReg] using h.keyIndex


theorem backward_second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (start stop : Nat)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardDiagonalReady M c s t start acc) :
    ∃ final, FP.Structural.range (withDot M D) "i" start stop c.BTS (backwardSecondBody c) t = some final ∧
      BackwardReady M c s final (foldRange (backwardUpdate M D c.toBackwardConfig s Bool.true) start stop c.BTS acc) := by
  apply range_values (withDot M D) "i" (backwardSecondBody c)
    (backwardUpdate M D c.toBackwardConfig s Bool.true) (fun n a t => BackwardDiagonalReady M c s t n a)
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
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
