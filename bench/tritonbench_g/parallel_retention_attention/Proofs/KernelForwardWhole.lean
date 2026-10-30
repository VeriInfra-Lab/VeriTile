/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardLoop
import bench.tritonbench_g.fused_recurrent_hgrn.Sources.Mathematical_FPStages
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot ForwardConfig queryChunk keyChunk valueChunk foldRange range_values)
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def firstPrefix (c : ForwardConfig) : List ComputeStmt := (forwardKernel c).surfaceBody.take 16
def betweenLoops (c : ForwardConfig) : List ComputeStmt := ((forwardKernel c).surfaceBody.drop 17).take 7
def forwardSuffix (c : ForwardConfig) : List ComputeStmt := (forwardKernel c).surfaceBody.drop 25

def initialAccumulator {α : Type} (M : Algebra α) (c : ForwardConfig) : (TileIndex [c.BTL,c.BV] → α) :=
  fun _ => M.literal (some .fp32) .real 0

theorem first_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ final, run (withDot M D) (firstPrefix c) s = some final ∧
      ForwardReady M c s final 0 (initialAccumulator M c) := by
  simp [firstPrefix, forwardKernel, forwardSource, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType,
    numeric, bop, Region.cast, withDot, TileShape.indexToList,
    BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets, ofFloat, toFloat, logDecay, rowDecay]
  constructor <;> simp [State.setReg, forwardQuery, initialAccumulator, matrixRead,
    keyPointer, valuePointer, keyChunk, valueChunk, queryChunk, bop, numeric,
    ofFloat, toFloat, withDot, logDecay, rowDecay]
  all_goals rfl

theorem between_loops_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (last : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t last acc) :
    ∃ final, run (withDot M D) (betweenLoops c) t = some final ∧
      DiagonalReady M c s final (queryChunk c s) (betweenValues M c s acc) := by
  simp [betweenLoops, forwardKernel, forwardSource, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType,
    numeric, bop, Region.cast, withDot, h.output, h.decayLog, h.head, h.chunk, h.keyBlock, h.valueBlock,
    TileShape.indexToList, ofFloat, toFloat]
  constructor
  · constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.query, h.output,
      h.decayLog, h.decay, h.chunk, h.head, h.keyBlock, h.valueBlock,
      keyPointer, valuePointer, keyChunk, valueChunk, queryChunk, betweenValues, rowDecay, bop]
    all_goals rfl
  · rfl
  · simp [State.setReg]
  · simp [State.setReg, queryChunk]

theorem forward_suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (last : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t last acc) :
    ∃ final, run (withDot M D) (forwardSuffix c) t = some final ∧
      final.mem = (outputMemory M c s acc).mem := by
  simp [forwardSuffix, forwardKernel, forwardSource, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType, store,
    numeric, bop, natLt, numericLt, Region.cast, withDot,
    h.head, h.chunk, h.keyBlock, h.valueBlock, h.output,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, outputMemory, queryChunk, valueChunk, Nat.add_assoc]
  apply scatter_memory_congr
  exact h.mem


def firstRangeStatement (c : ForwardConfig) : ComputeStmt :=
  .forRangeDyn "_i" (.constNat 0)
    (.mul .nat .nil (.ref .nat [] "i_c") (.constNat c.BTL)) (.constNat c.BTS)
    (firstBody c.BTL c.BTS c.BK c.BV)

def secondRangeStatement (c : ForwardConfig) : ComputeStmt :=
  .forRangeDyn "_i"
    (.mul .nat .nil (.ref .nat [] "i_c") (.constNat c.BTL))
    (.mul .nat .nil (.add .nat .nil (.ref .nat [] "i_c") (.constNat 1)) (.constNat c.BTL))
    (.constNat c.BTS) (secondBody c.BTL c.BTS c.BK c.BV)

theorem forward_source_split (c : ForwardConfig) :
    (forwardKernel c).surfaceBody = firstPrefix c ++ [firstRangeStatement c] ++
      betweenLoops c ++ [secondRangeStatement c] ++ forwardSuffix c := by
  rfl

theorem first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (n : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t n acc) :
    step (withDot M D) (firstRangeStatement c) t =
      FP.Structural.range (withDot M D) "_i" 0 (queryChunk c s) c.BTS (firstBody c.BTL c.BTS c.BK c.BV) t := by
  simp [firstRangeStatement, step, evalOp_unfold, boundaryZero, h.chunk, numeric, bop, Option.bind, queryChunk]

theorem second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (n : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t n acc) :
    step (withDot M D) (secondRangeStatement c) t =
      FP.Structural.range (withDot M D) "_i" (queryChunk c s) ((s.pids 1+1)*c.BTL)
        c.BTS (secondBody c.BTL c.BTS c.BK c.BV) t := by
  simp [secondRangeStatement, step, evalOp_unfold, boundaryZero, h.chunk, numeric, bop, Option.bind, queryChunk]

/-- The complete two-loop forward source computes the independently defined
ordered O recurrence and its exact memory frame. No final-result, real
arithmetic, dot expansion or reduction-law premise is used. -/
specification forward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (forwardKernel c) s = some final ∧
      final.mem = (outputMemory M c s (forwardValues M D c s)).mem := by
  obtain ⟨initial, hp, hi⟩ := first_prefix_run M D c s
  obtain ⟨first, hr, last, hf⟩ := first_range_run M D c s initial 0 (queryChunk c s) _ hi
  obtain ⟨diagonal, hm, hd⟩ := between_loops_run M D c s first last _ hf
  obtain ⟨second, hs, finish, hz⟩ := second_range_run M D c s diagonal (queryChunk c s)
    ((s.pids 1+1)*c.BTL) _ hd
  obtain ⟨final, he, hout⟩ := forward_suffix_run M D c s second finish _ hz
  refine ⟨final, ?_, ?_⟩
  · have hexec : FP.Structural.exec (withDot M D) (forwardKernel c) s =
        run (withDot M D) (forwardKernel c).surfaceBody s := by
      cases forwardKernel c
      rfl
    rw [hexec, forward_source_split]
    simp only [run_append, hp, bind, pure, Option.bind, run, first_range_statement M D c s initial 0 _ hi,
      hr, hm, second_range_statement M D c s diagonal (queryChunk c s) _ hd.toForwardReady,
      hs, he]
  · simpa only [forwardValues, initialAccumulator] using hout

#axiomsClean forward_source_split
#axiomsClean forward_fp32

#axiomsClean first_prefix_run
#axiomsClean between_loops_run
#axiomsClean forward_suffix_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
