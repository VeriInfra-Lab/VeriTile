import bench.tritonbench_g.parallel_attention.Proofs.CallerQueryLoop

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def queryPrefix (c : QueryConfig) := ((kernel c).surfaceBody.drop 5).take 11
def queryMiddle (c : QueryConfig) := ((kernel c).surfaceBody.drop 17).take 5
def querySuffix (c : QueryConfig) := ((kernel c).surfaceBody.drop 23).take 2
def queryFirstRange (c : QueryConfig) : ComputeStmt :=
  ((kernel c).surfaceBody[16]?).getD (.ifThen (.constBool Bool.false) [])
def querySecondRange (c : QueryConfig) : ComputeStmt :=
  ((kernel c).surfaceBody[22]?).getD (.ifThen (.constBool Bool.false) [])

def queryCode (c : QueryConfig) : List ComputeStmt := ((kernel c).surfaceBody.drop 5).take 20

theorem QueryReady.registers {α : Type} [Inhabited α] {M : Algebra α} {c : QueryConfig}
    {s t : State α} {start : Nat} {acc : TileIndex [c.BTL,c.BK] → α}
    (h : QueryReady M c s t start acc) : Registers c.toBackwardConfig t :=
  ⟨h.head,h.chunk,h.keyBlock,h.valueBlock⟩

theorem query_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) (r : Registers c.toBackwardConfig s) :
    ∃ final, run (withDot M D) (queryPrefix c) s = some final ∧
      QueryReady M c s final 0 (fun _ => M.literal (some .fp32) .real 0) := by
  simp [queryPrefix, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    r.head,r.chunk,r.keyBlock,r.valueBlock,ComputeDType.eraseDType, numeric, bop, Region.cast, withDot,
    TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    ofFloat, toFloat, natLt, numericLt]
  constructor <;> simp [r.head,r.chunk,r.keyBlock,r.valueBlock, queryInput, queryOutGrad, queryNormGrad,
    matrixRead, rawMatrixRead, queryKeyPointer, queryValuePointer, bop, numeric, ofFloat, toFloat]
  all_goals rfl

theorem query_middle_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (last : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryReady M c s t last acc) :
    ∃ final, run (withDot M D) (queryMiddle c) t = some final ∧
      QueryDiagonalReady M c s final (c.i_c*c.BTL) (queryScale M c acc) := by
  simp [queryMiddle, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, Region.cast, withDot, h.head,h.chunk,h.keyBlock,h.valueBlock,h.queryGrad, ofFloat, toFloat]
  constructor
  · constructor <;> simp [h.head,h.chunk,h.keyBlock,h.valueBlock, h.mem, h.pids, h.undef, h.query, h.outGrad,
      h.normGrad, queryScale, queryKeyPointer, queryValuePointer]
    all_goals rfl
  · rfl
  · simp [State.setReg]
  · simp [State.setReg]

theorem query_suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s t : State α) (last : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : QueryReady M c s t last acc) :
    ∃ final, run (withDot M D) (querySuffix c) t = some final ∧
      final.mem = (queryOutputMemory M c s acc).mem ∧ Registers c.toBackwardConfig final := by
  simp [querySuffix, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, Region.cast, withDot, store,
    h.head,h.chunk,h.keyBlock,h.valueBlock,h.queryGrad, TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, ofFloat, toFloat, queryOutputMemory]
  constructor
  · apply scatter_memory_congr
    exact h.mem
  · constructor <;> simp [h.head,h.chunk,h.keyBlock,h.valueBlock]

theorem query_source_split (c : QueryConfig) :
    queryCode c = queryPrefix c ++ [queryFirstRange c] ++
      queryMiddle c ++ [querySecondRange c] ++ querySuffix c := rfl

theorem query_first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (t : State α) (r : Registers c.toBackwardConfig t) :
    step (withDot M D) (queryFirstRange c) t =
      FP.Structural.range (withDot M D) "_" 0 (c.i_c*c.BTL) c.BTS (queryFirstBody c) t := by
  simp [queryFirstRange, queryFirstBody, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    step, evalOp_unfold, numeric, bop, Option.bind,r.head,r.chunk,r.keyBlock,r.valueBlock]

theorem query_second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (t : State α) (r : Registers c.toBackwardConfig t) :
    step (withDot M D) (querySecondRange c) t =
      FP.Structural.range (withDot M D) "_" (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS (querySecondBody c) t := by
  simp [querySecondRange, querySecondBody, kernel, shellSource, backwardPastedKernel, ComputeKernel.surfaceBody,
    step, evalOp_unfold, numeric, bop, Option.bind,r.head,r.chunk,r.keyBlock,r.valueBlock]

/-- Complete DQ source execution, including both loops, intermediate scaling,
all casts and the per-value-shard output store. No execution-success premise. -/
specification query_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) (r : Registers c.toBackwardConfig s) :
    ∃ final, run (withDot M D) (queryCode c) s = some final ∧
      final.mem = (queryOutputMemory M c s (queryValues M D c s)).mem ∧ Registers c.toBackwardConfig final := by
  obtain ⟨initial, hp, hi⟩ := query_prefix_run M D c s r
  obtain ⟨first, hr, last, hf⟩ := query_first_range_run M D c s initial 0 (c.i_c*c.BTL) _ hi
  obtain ⟨diagonal, hm, hd⟩ := query_middle_run M D c s first last _ hf
  obtain ⟨second, hs, finish, hz⟩ := query_second_range_run M D c s diagonal
    (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd
  obtain ⟨final, he, hout, registers⟩ := query_suffix_run M D c s second finish _ hz
  refine ⟨final, ?_, ?_, registers⟩
  · have hfirst:=query_first_range_statement M D c initial hi.registers
    have hsecond:=query_second_range_statement M D c diagonal hd.toQueryReady.registers
    rw [query_source_split]
    simp only [run_append,hp,bind,pure,Option.bind,run,hfirst,hr,hm,hsecond,hs,he]

  · simpa only [queryValues] using hout

#axiomsClean query_prefix_run
#axiomsClean query_middle_run
#axiomsClean query_suffix_run
#axiomsClean query_source_split
#axiomsClean query_fp32
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
