/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.ForwardPrefix
import bench.tritonbench_g.parallel_retention_attention.Proofs.IndexBounds
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Sources.Mathematical_FPStages
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot ForwardConfig queryChunk keyChunk valueChunk foldRange)
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false


def betweenLoops (c : ForwardConfig) : List ComputeStmt := ((forwardKernel c).surfaceBody.drop 33).take 7
def forwardSuffix (c : ForwardConfig) : List ComputeStmt := (forwardKernel c).surfaceBody.drop 41

theorem between_loops_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (last : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t last acc) (bounds : ForwardIndexBounds c s) :
    ∃ final, run (withDot M D) (betweenLoops c) t = some final ∧
      DiagonalReady M c s final (queryChunk c s) (betweenValues M c s acc) := by
  simp [betweenLoops, forwardKernel, forwardSource,forwardPastedKernel, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType,
    numeric, bop, numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,IntegerExpr.sameShape,Region.cast,withDot, h.output, h.decayLog, h.head, h.chunk, h.keyBlock, h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,
    TileShape.indexToList,ofFloat,toFloat,Index32.nat_remainder,bounds.query_cast,bounds.key_cast,bounds.value_cast,bounds.key_cast_eval,bounds.value_cast_eval,Index32.natural_product]
  constructor
  · constructor <;> simp [h.mem, h.pids, h.undef, h.query, h.output,
      h.decayLog, h.decay, h.chunk, h.head, h.keyBlock, h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,
      keyPointer, valuePointer, keyChunk, valueChunk, queryChunk, betweenValues, rowDecay, bop]
    all_goals rfl
  · rfl
  · simp [State.setReg]
  · simp [queryChunk]

theorem forward_suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (last : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t last acc) (bounds : ForwardIndexBounds c s) :
    ∃ final, run (withDot M D) (forwardSuffix c) t = some final ∧
      final.mem = (outputMemory M c s acc).mem := by
  simp [forwardSuffix, forwardKernel, forwardSource,forwardPastedKernel, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind, ComputeDType.eraseDType, store,
    numeric, bop, natLt, numericLt, Region.cast, withDot,
    h.head, h.chunk, h.keyBlock, h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8, h.output,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,outputMemory,queryChunk,valueChunk,Nat.add_assoc,Index32.nat_remainder,bounds.query_cast,bounds.key_cast,bounds.value_cast,bounds.key_cast_eval,bounds.value_cast_eval,Index32.natural_product]
  apply scatter_memory_congr
  exact h.mem


def firstRangeStatement (c : ForwardConfig) : ComputeStmt :=
  ((forwardKernel c).surfaceBody[32]?).getD (.ifThen (.constBool Bool.false) [])

def secondRangeStatement (c : ForwardConfig) : ComputeStmt :=
  ((forwardKernel c).surfaceBody[40]?).getD (.ifThen (.constBool Bool.false) [])

theorem forward_source_split (c : ForwardConfig) :
    (forwardKernel c).surfaceBody = firstPrefix c ++ [firstRangeStatement c] ++
      betweenLoops c ++ [secondRangeStatement c] ++ forwardSuffix c := by
  rfl

theorem first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (n : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t n acc) :
    step (withDot M D) (firstRangeStatement c) t =
      FP.Structural.rangeInt (withDot M D) "_" 0 (queryChunk c s : Int) c.BTS (firstBody c.BTL c.BTS c.BK c.BV) t := by
  simp [firstRangeStatement,forwardKernel,forwardSource,forwardPastedKernel,ComputeKernel.surfaceBody,firstBody,step, evalOp_unfold, boundaryZero, h.chunk,numeric,bop,Option.bind,queryChunk,Int.natCast_mul,Int.natCast_add]

theorem second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (n : Nat) (acc : (TileIndex [c.BTL,c.BV] → α))
    (h : ForwardReady M c s t n acc) :
    step (withDot M D) (secondRangeStatement c) t =
      FP.Structural.rangeInt (withDot M D) "_" (queryChunk c s : Int) (((s.pids 1+1)*c.BTL : Nat) : Int)
        c.BTS (secondBody c.BTL c.BTS c.BK c.BV) t := by
  simp [secondRangeStatement,forwardKernel,forwardSource,forwardPastedKernel,ComputeKernel.surfaceBody,secondBody,step, evalOp_unfold, boundaryZero, h.chunk,numeric,bop,Option.bind,queryChunk,Int.natCast_mul,Int.natCast_add]

/-- The complete two-loop forward source computes the independently defined
ordered O recurrence and its exact memory frame. No final-result, real
arithmetic, dot expansion or reduction-law premise is used. -/
specification forward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) (bounds : ForwardIndexBounds c s) :
    ∃ final, FP.Structural.exec (withDot M D) (forwardKernel c) s = some final ∧
      final.mem = (outputMemory M c s (forwardValues M D c s)).mem := by
  obtain ⟨initial, hp, hi⟩ := first_prefix_run M D c s bounds
  obtain ⟨first, hr, last, hf⟩ := first_range_run M D c s initial 0 (queryChunk c s) _ hi
  have hrZero : FP.Structural.rangeInt (withDot M D) "_" 0 (queryChunk c s : Int) c.BTS
      (firstBody c.BTL c.BTS c.BK c.BV) initial=some first := by
    simpa only [Int.natCast_zero] using hr
  obtain ⟨diagonal, hm, hd⟩ := between_loops_run M D c s first last _ hf bounds
  obtain ⟨second, hs, finish, hz⟩ := second_range_run M D c s diagonal (queryChunk c s)
    ((s.pids 1+1)*c.BTL) _ hd bounds.lanes
  obtain ⟨final, he, hout⟩ := forward_suffix_run M D c s second finish _ hz bounds
  refine ⟨final, ?_, ?_⟩
  · have hexec : FP.Structural.exec (withDot M D) (forwardKernel c) s =
        run (withDot M D) (forwardKernel c).surfaceBody s := by
      cases forwardKernel c
      rfl
    rw [hexec, forward_source_split]
    simp only [run_append, hp, bind, pure, Option.bind, run, Int.ofNat_zero, first_range_statement M D c s initial 0 _ hi,
      hrZero, hm, second_range_statement M D c s diagonal (queryChunk c s) _ hd.toForwardReady,
      hs, he]
  · simpa only [forwardValues, initialAccumulator] using hout

#axiomsClean forward_source_split
#axiomsClean forward_fp32

#axiomsClean first_prefix_run
#axiomsClean between_loops_run
#axiomsClean forward_suffix_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
