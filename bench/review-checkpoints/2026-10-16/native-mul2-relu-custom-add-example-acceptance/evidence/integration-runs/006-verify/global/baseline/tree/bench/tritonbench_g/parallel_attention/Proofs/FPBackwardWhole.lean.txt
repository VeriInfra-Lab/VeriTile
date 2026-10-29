import bench.tritonbench_g.parallel_attention.Proofs.FPBackwardLoop
import bench.tritonbench_g.parallel_attention.Proofs.FPForwardWhole
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def backwardPrefix (c : BackwardConfig) := (backwardKernel c).surfaceBody.take 6
def backwardMiddle (c : BackwardConfig) := ((backwardKernel c).surfaceBody.drop 7).take 3
def backwardSuffix (c : BackwardConfig) := (backwardKernel c).surfaceBody.drop 11
def backwardFirstRange (c : BackwardConfig) : ComputeStmt :=
  ((backwardKernel c).surfaceBody[6]?).getD (.ifThen (.constBool Bool.false) [])
def backwardSecondRange (c : BackwardConfig) : ComputeStmt :=
  ((backwardKernel c).surfaceBody[10]?).getD (.ifThen (.constBool Bool.false) [])
def backwardInitial {α : Type} (M : Algebra α) (c : BackwardConfig) : BackwardAccumulator α c.BTL c.BK c.BV :=
  ⟨fun _ => M.literal (some .fp32) .real 0, fun _ => M.literal (some .fp32) .real 0⟩

theorem backward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) :
    ∃ final, run (withDot M D) (backwardPrefix c) s = some final ∧
      BackwardReady M c s final (backwardInitial M c) := by
  simp [backwardPrefix, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, Region.cast, withDot,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, ofFloat, toFloat]
  constructor <;> simp [State.setReg, backwardKey, backwardValue, matrixRead, backwardInitial]
  all_goals rfl

theorem backward_middle_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    ∃ final, run (withDot M D) (backwardMiddle c) t = some final ∧
      BackwardDiagonalReady M c s final (c.i_c*c.BTL) acc := by
  simp [backwardMiddle, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType]
  constructor
  · constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad]
  · rfl
  · simp [State.setReg]
  · simp [State.setReg]

theorem backward_suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    ∃ final, run (withDot M D) (backwardSuffix c) t = some final ∧
      final.mem = (backwardOutputMemory M c s acc).mem := by
  simp [backwardSuffix, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, Region.cast, withDot, store,
    h.keyGrad, h.valueGrad, TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, ofFloat, toFloat, backwardOutputMemory]
  apply scatter_memory_congr
  apply scatter_memory_congr
  exact h.mem

theorem backward_source_split (c : BackwardConfig) :
    (backwardKernel c).surfaceBody = backwardPrefix c ++ [backwardFirstRange c] ++
      backwardMiddle c ++ [backwardSecondRange c] ++ backwardSuffix c := by rfl

theorem backward_first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (t : State α) :
    step (withDot M D) (backwardFirstRange c) t =
      FP.Structural.range (withDot M D) "j" 0 (descendingCount c) 1 (backwardFirstBody c) t := by
  simp [backwardFirstRange, backwardFirstBody, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    step, evalOp_unfold, numeric, bop, Option.bind, descendingCount]

theorem backward_second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (t : State α) :
    step (withDot M D) (backwardSecondRange c) t =
      FP.Structural.range (withDot M D) "i" (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS (backwardSecondBody c) t := by
  simp [backwardSecondRange, backwardSecondBody, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    step, evalOp_unfold, numeric, bop, Option.bind]

/-- Complete original accepted DK/DV stage, including both loop ranges and
all loads, casts, dot options, branch orders and output memory framing. -/
specification backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (backwardKernel c) s = some final ∧
      final.mem = (backwardOutputMemory M c s (backwardValues M D c s)).mem := by
  obtain ⟨initial, hp, hi⟩ := backward_prefix_run M D c s
  obtain ⟨first, hr, hf⟩ := backward_first_range_run M D c s initial 0 (descendingCount c) _ hi
  obtain ⟨diagonal, hm, hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second, hs, hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd
  obtain ⟨final, he, hout⟩ := backward_suffix_run M D c s second _ hz
  refine ⟨final, ?_, ?_⟩
  · have hexec : FP.Structural.exec (withDot M D) (backwardKernel c) s =
        run (withDot M D) (backwardKernel c).surfaceBody s := by cases backwardKernel c; rfl
    rw [hexec, backward_source_split]
    simp only [run_append, hp, bind, pure, Option.bind, run, backward_first_range_statement,
      hr, hm, backward_second_range_statement, hs, he]
  · simpa only [backwardValues, backwardInitial] using hout

#axiomsClean backward_fp32
#axiomsClean backward_source_split
#axiomsClean backward_prefix_run
#axiomsClean backward_middle_run
#axiomsClean backward_suffix_run
end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
