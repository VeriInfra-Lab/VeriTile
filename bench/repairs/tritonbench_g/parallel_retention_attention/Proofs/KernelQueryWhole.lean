/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryLoop
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPStages

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig foldRange)
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
open FPStages (logDecay rowDecay blockDecay)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def prefixCode (c : BackwardConfig) (dq : RegionName) := (source c dq).surfaceBody.take 8
def middle (c : BackwardConfig) (dq : RegionName) := ((source c dq).surfaceBody.drop 9).take 5
def suffix (c : BackwardConfig) (dq : RegionName) := (source c dq).surfaceBody.drop 15
def firstRange (c : BackwardConfig) : ComputeStmt :=
  .forRange "_i" 0 (c.i_c*c.BTL) c.BTS (firstBody c)
def secondRange (c : BackwardConfig) : ComputeStmt :=
  .forRange "_i" (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS (secondBody c)

theorem source_split (c : BackwardConfig) (dq : RegionName) :
    (source c dq).surfaceBody = prefixCode c dq ++ [firstRange c] ++ middle c dq ++
      [secondRange c] ++ suffix c dq := by rfl

theorem source_supported (c : BackwardConfig) (dq : RegionName) :
    ∃ alg, (source c dq).toAlgorithm? = Except.ok alg := by
  simp [source, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, pure]

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    ∃ final, run (withDot M D) (prefixCode c dq) s = some final ∧
      Ready M c s final 0 (fun _ => M.literal (some .fp32) .real 0) := by
  simp [prefixCode, source, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType,
    numeric, bop, Region.cast, withDot, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, ofFloat, toFloat, logDecay, rowDecay, blockDecay]
  constructor <;> simp [State.setReg, outputGradient, matrixRead,
    keyPointer, valuePointer, bop, numeric, ofFloat, toFloat, withDot, logDecay, rowDecay, blockDecay]
  all_goals rfl

theorem middle_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s t : State α) (last : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : Ready M c s t last acc) :
    ∃ final, run (withDot M D) (middle c dq) t = some final ∧
      DiagonalReady M c s final (c.i_c*c.BTL) (betweenValues M c acc) := by
  simp [middle, source, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType,
    numeric, bop, Region.cast, withDot, h.queryGrad, h.decayLog, TileShape.indexToList, ofFloat, toFloat]
  constructor
  · constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.outGrad, h.queryGrad,
      h.decayLog, h.decayBlock, h.decay, keyPointer, valuePointer, betweenValues, rowDecay, bop]
    all_goals rfl
  · rfl
  · simp [State.setReg]
  · simp [State.setReg]

theorem outputMemory_undef {α : Type} (M : Algebra α) (c : BackwardConfig) (dq : RegionName)
    (s : State α) (acc : TileIndex [c.BTL,c.BK] → α) :
    (outputMemory M c dq s acc).undef = s.undef := by
  unfold outputMemory
  generalize TileShape.allIndices [c.BTL,c.BK] = indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

theorem suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s t : State α) (last : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : Ready M c s t last acc) :
    ∃ final, run (withDot M D) (suffix c dq) t = some final ∧
      final.mem = (outputMemory M c dq s acc).mem ∧ final.undef = s.undef := by
  simp [suffix, source, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType, store,
    numeric, bop, natLt, numericLt, Region.cast, withDot, h.queryGrad,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat]
  constructor
  · apply scatter_memory_congr
    exact h.mem
  · change (outputMemory M c dq _ acc).undef = s.undef
    rw [outputMemory_undef]
    exact h.undef

theorem first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (t : State α) :
    step (withDot M D) (firstRange c) t =
      FP.Structural.range (withDot M D) "_i" 0 (c.i_c*c.BTL) c.BTS (firstBody c) t := by
  simp [firstRange, step, evalOp_unfold, boundaryZero, Option.bind]

theorem second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (t : State α) :
    step (withDot M D) (secondRange c) t =
      FP.Structural.range (withDot M D) "_i" (c.i_c*c.BTL) ((c.i_c+1)*c.BTL)
        c.BTS (secondBody c) t := by
  simp [secondRange, step, evalOp_unfold, boundaryZero, Option.bind]

/-- Complete DQ execution and full memory result for arbitrary initial states,
dimensions, strides and aliases in the source model. Native legality and the
numerical primitive binding are separate; no output-equality premise is used. -/
theorem query_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (source c dq) s = some final ∧
      final.mem = (outputMemory M c dq s (values M D c s)).mem ∧ final.undef = s.undef := by
  obtain ⟨initial, hp, hi⟩ := prefix_run M D c dq s
  obtain ⟨first, hr, last, hf⟩ := first_range_run M D c s initial 0 (c.i_c*c.BTL) _ hi
  obtain ⟨diagonal, hm, hd⟩ := middle_run M D c dq s first last _ hf
  obtain ⟨second, hs, finish, hz⟩ := second_range_run M D c s diagonal
    (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd
  obtain ⟨final, he, hout, hu⟩ := suffix_run M D c dq s second finish _ hz
  refine ⟨final, ?_, ?_, hu⟩
  · have hexec : FP.Structural.exec (withDot M D) (source c dq) s =
        run (withDot M D) (source c dq).surfaceBody s := by cases source c dq; rfl
    rw [hexec, source_split]
    simp only [run_append, hp, bind, pure, Option.bind, run, first_range_statement,
      hr, hm, second_range_statement, hs, he]
  · simpa only [values] using hout

#axiomsClean source_split
#axiomsClean source_supported
#axiomsClean prefix_run
#axiomsClean middle_run
#axiomsClean suffix_run
#axiomsClean query_fp32
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
