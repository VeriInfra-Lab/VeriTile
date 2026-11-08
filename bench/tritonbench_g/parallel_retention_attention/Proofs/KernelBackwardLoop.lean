/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardGraph
import bench.tritonbench_g.parallel_retention_attention.Proofs.SignedRange
import bench.tritonbench_g.parallel_retention_attention.Proofs.DescendingRange
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig BackwardAccumulator matrixTranspose descendingCount descendingStart foldRange backwardOutputMemory)

noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def backwardFirstBody (c : BackwardConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (backwardKernel c).surfaceBody
  match code[43]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []
def backwardSecondBody (c : BackwardConfig) : List ComputeStmt :=
  let code : List ComputeStmt := (backwardKernel c).surfaceBody
  match code[51]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
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

  argument0 : t.regs .int [] "__veritile_argument_16973089353068028100_0"=some (fun _=>(c.i_bh : Int))
  argument1 : t.regs .int [] "__veritile_argument_16973089353068028100_1"=some (fun _=>(c.i_c : Int))
  argument2 : t.regs .int [] "__veritile_argument_16973089353068028100_2"=some (fun _=>(c.i_k : Int))
  argument3 : t.regs .int [] "__veritile_argument_16973089353068028100_3"=some (fun _=>(c.i_v : Int))
  argument4 : t.regs .int [] "__veritile_argument_16973089353068028100_4"=some (fun _=>(c.i_h : Int))
  argument5 : t.regs .int [] "__veritile_argument_16973089353068028100_5"=some (fun _=>(c.s_qk_h : Int))
  argument6 : t.regs .int [] "__veritile_argument_16973089353068028100_6"=some (fun _=>(c.s_qk_t : Int))
  argument7 : t.regs .int [] "__veritile_argument_16973089353068028100_7"=some (fun _=>(c.s_qk_d : Int))
  argument8 : t.regs .int [] "__veritile_argument_16973089353068028100_8"=some (fun _=>(c.s_vo_h : Int))
  argument9 : t.regs .int [] "__veritile_argument_16973089353068028100_9"=some (fun _=>(c.s_vo_t : Int))
  argument10 : t.regs .int [] "__veritile_argument_16973089353068028100_10"=some (fun _=>(c.s_vo_d : Int))
  argument11 : t.regs .int [] "__veritile_argument_16973089353068028100_11"=some (fun _=>(c.B : Int))
  argument12 : t.regs .int [] "__veritile_argument_16973089353068028100_12"=some (fun _=>(c.H : Int))
  argument13 : t.regs .int [] "__veritile_argument_16973089353068028100_13"=some (fun _=>(c.T : Int))

theorem backward_first_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hj : t.regs .int [] "__rev_i" = some (fun _ => -(start : Int)))
    (bounds : HelperIndexBounds c) (timeBound : start<2147483648) :
    ∃ final, run (withDot M D) (backwardFirstBody c) t = some final ∧
      BackwardReady M c s final (backwardUpdate M D c s Bool.false start acc) := by
  simp [ backwardFirstBody, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13, hj, h.mem, h.pids, h.undef,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt,bounds.key_cast,bounds.value_cast,
    Index32.nat_value start timeBound,Index32.natural_product]
  all_goals constructor <;> simp [h.mem, h.pids, h.undef, h.key, h.value, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,
    backwardUpdate, descendingStart, matrixRead, matrixTranspose, transposeValues, rowDecay, blockDecay, decayedKey,
    bop, numeric, ofFloat, toFloat,bounds.key_cast,bounds.value_cast,
    Index32.nat_value start timeBound,Index32.natural_product]
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
    (hi : t.regs .int [] "i" = some (fun _ => (start : Int)))
    (bounds : HelperIndexBounds c) (timeBound : start<2147483648) :
    ∃ final, run (withDot M D) (backwardSecondBody c) t = some final ∧
      BackwardDiagonalReady M c s final (start+c.BTS) (backwardUpdate M D c s Bool.true start acc) := by
  have lane (i : Fin c.BTL) : Index32.signed (i.val : Int)=(i.val : Int):=
    Index32.nat_value i.val (Nat.lt_of_lt_of_le i.isLt bounds.lanes)
  simp [ backwardSecondBody, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, numericEq, Region.cast, withDot,
    h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13, hi, h.mem, h.pids, h.undef, h.queryIndex, h.keyIndex,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt, natLe, numericLe,bounds.key_cast,bounds.value_cast,
    Index32.nat_value start timeBound,Index32.natural_product]
  constructor
  · constructor <;> simp [h.mem,h.pids,h.undef,h.key,h.value,h.decayLog,h.decayBlock,
      h.decayHead,h.decayQuery,h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,
      backwardUpdate,matrixRead,matrixTranspose,transposeValues,rowDecay,blockDecay,decayedKey,
      bop,numeric,ofFloat,toFloat,bounds.key_cast,bounds.value_cast,
      Index32.nat_value start timeBound,Index32.natural_product]
    all_goals simp only [Index32.expansion (α := α),lane,←Int.natCast_add]
    all_goals rfl
  · have := h.lower
    omega
  · simp [h.queryIndex,h.keyIndex,bop,numeric]
    funext i
    have := h.lower
    change start-c.i_c*c.BTL+i.1.val+c.BTS=start+c.BTS-c.i_c*c.BTL+i.1.val
    omega
  · simp [h.keyIndex]


theorem backwardReady_j {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (index : Int) :
    BackwardReady M c s (t.setReg "__rev_i" .int [] (fun _ => index)) acc := by
  constructor <;> simp [h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13]

theorem backward_first_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start : Nat) (bound : start≤descendingCount c)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardReady M c s t acc) (bounds : BackwardIndexBounds c) :
    ∃ final, rangeInt (withDot M D) "__rev_i"
        (DescendingRange.lower c+(start : Int)*(c.BTS : Int)) (DescendingRange.upper c) c.BTS
        (backwardFirstBody c) t = some final ∧
      BackwardReady M c s final
        (foldRange (fun j => backwardUpdate M D c s Bool.false (descendingStart c j))
          start (descendingCount c) 1 acc) := by
  by_cases zero : c.BTS=0
  · have count : descendingCount c=0:=by simp [descendingCount,zero]
    have first : start=0:=by omega
    subst start
    exact ⟨t,by simp [rangeInt,zero],by simpa [foldRange,count] using h⟩
  have positive : 0<c.BTS:=by omega
  by_cases live : start<descendingCount c
  · have current : DescendingRange.lower c+(start : Int)*(c.BTS : Int)=
        -(descendingStart c start : Int) := by
      have equal:=congrArg Neg.neg (DescendingRange.source_index c positive start live)
      simpa only [neg_neg] using equal
    obtain ⟨middle,ran,maintained⟩:=backward_first_body_run M D c s _ (descendingStart c start) acc
      (backwardReady_j M c s t acc h (-(descendingStart c start : Int))) (by simp [State.setReg]) bounds.toHelperIndexBounds (bounds.descending_lt start)
    obtain ⟨final,finished,invariant⟩:=backward_first_range_run M D c s middle (start+1) (by omega) _ maintained bounds
    refine ⟨final,?_,?_⟩
    · rw [rangeInt]
      simp only [zero,↓reduceIte,(DescendingRange.source_active c positive start).mpr live]
      rw [current,ran]
      have next : DescendingRange.lower c+(start : Int)*(c.BTS : Int)+(c.BTS : Int)=
          DescendingRange.lower c+((start+1 : Nat) : Int)*(c.BTS : Int) := by
        simp [Nat.cast_add,add_mul,add_assoc]
      rw [←current,next]
      exact finished
    · rw [foldRange]
      simpa only [Nat.one_ne_zero,live,↓reduceIte] using invariant
  · refine ⟨t,?_,?_⟩
    · rw [rangeInt]
      simp [zero,DescendingRange.source_active c positive start,live]
    · simpa [foldRange,live] using h
termination_by descendingCount c-start
decreasing_by omega

theorem backwardDiagonalReady_i {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s t : State α) (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc) (i : Nat) :
    BackwardDiagonalReady M c s (t.setReg "i" .int [] (fun _ => (i : Int))) start acc := by
  constructor
  · constructor <;> simp [h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13]
  · exact h.lower
  · simpa [State.setReg] using h.queryIndex
  · simpa [State.setReg] using h.keyIndex

theorem backward_second_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (start stop : Nat)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardDiagonalReady M c s t start acc) (bounds : HelperIndexBounds c)
    (stopBound : stop≤2147483648) :
    ∃ final, FP.Structural.rangeInt (withDot M D) "i" (start : Int) (stop : Int) c.BTS (backwardSecondBody c) t = some final ∧
      BackwardReady M c s final (foldRange (backwardUpdate M D c s Bool.true) start stop c.BTS acc) := by
  apply rangeInt_values (withDot M D) "i" (backwardSecondBody c)
    (backwardUpdate M D c s Bool.true) (fun n a t => BackwardDiagonalReady M c s t n a)
    (fun a t => BackwardReady M c s t a) start stop c.BTS acc t
  · intro n a u hu live
    exact backward_second_body_run M D c s _ n a (backwardDiagonalReady_i M c s u n a hu n) (by simp [State.setReg]) bounds (by omega)
  · intro n a u hu
    exact hu.toBackwardReady
  · exact h

#axiomsClean backward_first_body_run
#axiomsClean backward_second_body_run
#axiomsClean backward_first_range_run
#axiomsClean backward_second_range_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
