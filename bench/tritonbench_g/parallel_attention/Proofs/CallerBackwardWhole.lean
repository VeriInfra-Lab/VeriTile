import bench.tritonbench_g.parallel_attention.Proofs.CallerBackwardLoop

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def backwardPrefix (c : QueryConfig) := ((kernel c).surfaceBody.drop 26).take 10
def backwardMiddle (c : QueryConfig) := ((kernel c).surfaceBody.drop 37).take 5
def backwardSuffix (c : QueryConfig) := ((kernel c).surfaceBody.drop 43).take 4
def backwardFirstRange (c : QueryConfig) : ComputeStmt :=
  ((kernel c).surfaceBody[36]?).getD (.ifThen (.constBool Bool.false) [])
def backwardSecondRange (c : QueryConfig) : ComputeStmt :=
  ((kernel c).surfaceBody[42]?).getD (.ifThen (.constBool Bool.false) [])
def backwardInitial {α : Type} (M : Algebra α) (c : QueryConfig) : BackwardAccumulator α c.BTL c.BK c.BV :=
  ⟨fun _ => M.literal (some .fp32) .real 0, fun _ => M.literal (some .fp32) .real 0⟩

def backwardCode (c : QueryConfig) : List ComputeStmt := ((kernel c).surfaceBody.drop 26).take 21

theorem BackwardReady.registers {α : Type} [Inhabited α] {M : Algebra α} {c : QueryConfig}
    {s t : State α} {acc : BackwardAccumulator α c.BTL c.BK c.BV}
    (h : BackwardReady M c s t acc) : Registers c.toBackwardConfig t :=
  ⟨h.head,h.chunk,h.keyBlock,h.valueBlock⟩

theorem backward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) (r : Registers c.toBackwardConfig s) :
    ∃ final, run (withDot M D) (backwardPrefix c) s = some final ∧
      BackwardReady M c s final (backwardInitial M c) := by
  simp [backwardPrefix, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    r.head,r.chunk,r.keyBlock,r.valueBlock,ComputeDType.eraseDType, numeric, bop, Region.cast, withDot,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, ofFloat, toFloat]
  constructor <;> simp [r.head,r.chunk,r.keyBlock,r.valueBlock, backwardKey, backwardValue, matrixRead, backwardInitial]
  all_goals rfl

theorem backward_middle_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    ∃ final, run (withDot M D) (backwardMiddle c) t = some final ∧
      BackwardDiagonalReady M c s final (c.i_c*c.BTL) acc := by
  simp [backwardMiddle, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, Option.bind,h.head,h.chunk,h.keyBlock,h.valueBlock, ComputeDType.eraseDType]
  constructor
  · constructor <;> simp [h.head,h.chunk,h.keyBlock,h.valueBlock, h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad]
  · rfl
  · simp [h.head,h.chunk,h.keyBlock,h.valueBlock]
  · simp [h.head,h.chunk,h.keyBlock,h.valueBlock]

theorem backward_suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    ∃ final, run (withDot M D) (backwardSuffix c) t = some final ∧
      final.mem = (backwardOutputMemory M c.toBackwardConfig s acc).mem := by
  simp [backwardSuffix, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,h.head,h.chunk,h.keyBlock,h.valueBlock,
    ComputeDType.eraseDType, numeric, bop, Region.cast, withDot, store,
    h.keyGrad, h.valueGrad, TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, ofFloat, toFloat, backwardOutputMemory]
  apply scatter_memory_congr
  apply scatter_memory_congr
  exact h.mem

theorem backward_source_split (c : QueryConfig) :
    backwardCode c = backwardPrefix c ++ [backwardFirstRange c] ++
      backwardMiddle c ++ [backwardSecondRange c] ++ backwardSuffix c := by rfl

theorem backward_first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (t : State α) (r : Registers c.toBackwardConfig t) :
    step (withDot M D) (backwardFirstRange c) t =
      rangeInt (withDot M D) "__rev_i" (DescendingRange.lower c.toBackwardConfig) (DescendingRange.upper c.toBackwardConfig)
        c.BTS (backwardFirstBody c) t := by
  simp [backwardFirstRange,backwardFirstBody,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody,
    step,evalOp_unfold,boundaryZero,numeric,bop,Option.bind,r.head,r.chunk,r.keyBlock,r.valueBlock,DescendingRange.lower,DescendingRange.upper,sub_eq_add_neg,add_comm]

theorem backward_second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (t : State α) (r : Registers c.toBackwardConfig t) :
    step (withDot M D) (backwardSecondRange c) t =
      FP.Structural.range (withDot M D) "i" (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS (backwardSecondBody c) t := by
  simp [backwardSecondRange, backwardSecondBody, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    step, evalOp_unfold, boundaryZero, numeric, bop, Option.bind,r.head,r.chunk,r.keyBlock,r.valueBlock]

/-- Complete original accepted DK/DV stage, including both loop ranges and
all loads, casts, dot options, branch orders and output memory framing. -/
specification backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) (r : Registers c.toBackwardConfig s) :
    ∃ final, run (withDot M D) (backwardCode c) s = some final ∧
      final.mem = (backwardOutputMemory M c.toBackwardConfig s (backwardValues M D c.toBackwardConfig s)).mem := by
  obtain ⟨initial, hp, hi⟩ := backward_prefix_run M D c s r
  obtain ⟨first, hr, hf⟩ := backward_first_range_run M D c s initial 0 (by omega) _ hi
  simp only [Nat.cast_zero,zero_mul,add_zero] at hr
  obtain ⟨diagonal, hm, hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second, hs, hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd
  obtain ⟨final, he, hout⟩ := backward_suffix_run M D c s second _ hz
  refine ⟨final, ?_, ?_⟩
  · have hfirst:=backward_first_range_statement M D c initial hi.registers
    have hsecond:=backward_second_range_statement M D c diagonal hd.toBackwardReady.registers
    rw [backward_source_split]
    simp only [run_append,hp,bind,pure,Option.bind,run,hfirst,hr,hm,hsecond,hs,he]

  · simpa only [backwardValues, backwardInitial] using hout

#axiomsClean backward_fp32
#axiomsClean backward_source_split
#axiomsClean backward_prefix_run
#axiomsClean backward_middle_run
#axiomsClean backward_suffix_run
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
