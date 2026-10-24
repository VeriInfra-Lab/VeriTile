import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPPrefix
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 15000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def loopBody (c : Config) (columnPhase : Bool) : List ComputeStmt :=
  if columnPhase then columnBody c else blockBody c

def indexName (columnPhase : Bool) : RegName := if columnPhase then "start_n" else "block_index"

theorem column_key_function {α : Type} [Inhabited α] (M : Algebra α)
    (c : Config) (s : State α) (start : Nat) :
    keyValues M c s Bool.true start = fun (i : TileIndex [c.BD,c.BN]) =>
      if start+i.2.1.val < columnCount c s ∧ start < columnCount c s then
        (s.mem c.k (keyValueBase c s+i.1.val*c.skk+
          (if start < columnCount c s then
            (s.mem (Region.cast c.columns) (metadataIndex c s*c.NV+start+i.2.1.val)).read .nat*c.skn
           else 0))).read .fp16
      else M.cast none .real .fp16 (M.literal none .real 0.0) := by
  funext i
  simp [keyValues,keyActive,columnValues,ite_mul]

theorem column_value_function {α : Type} [Inhabited α] (M : Algebra α)
    (c : Config) (s : State α) (start : Nat) :
    valueValues M c s Bool.true start = fun (i : TileIndex [c.BN,c.BD]) =>
      if start+i.1.val < columnCount c s ∧ start < columnCount c s then
        (s.mem c.v (keyValueBase c s+i.2.1.val*c.svk+
          (if start < columnCount c s then
            (s.mem (Region.cast c.columns) (metadataIndex c s*c.NV+start+i.1.val)).read .nat*c.svn
           else 0))).read .fp16
      else M.cast none .real .fp16 (M.literal none .real 0.0) := by
  funext i
  simp [valueValues,keyActive,columnValues,ite_mul]

theorem loop_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (ph : Bool) (start : Nat) (a : Accumulator α c.BM c.BD)
    (h : Ready M c s t a) (hBN : 0<c.BN) :
    ∃ final, run (backend M D) (loopBody c ph)
      (t.setReg (indexName ph) .nat [] (fun _ => start)) = some final ∧
      Ready M c s final (update M D c s ph start a) := by
  cases ph <;>
    simp [loopBody,indexName,columnBody,blockBody,activeBody,parsedActiveBody,parsedKernel,source,
      maximumAssignment,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,
      integralFloorDiv,integralMod,backend,Region.cast,ofFloat,toFloat,State.setReg,
      h.seqlen,h.rows,h.lanes,h.keyptr,h.valueptr,h.outptr,h.numblocks,h.blockptr,
      h.numcolumns,h.columnptr,h.query,h.mask,h.maximum,h.normalizer,h.output,h.maxblocks,
      h.mem,h.pids,h.undef,TileShape.axisDim,hBN]
  all_goals constructor <;>
    try simp [State.setReg,h.seqlen,h.rows,h.lanes,h.keyptr,h.valueptr,h.outptr,h.numblocks,h.blockptr,
      h.numcolumns,h.columnptr,h.query,h.mask,h.maximum,h.normalizer,h.output,h.maxblocks,
      h.mem,h.pids,h.undef,update,keyValues,valueValues,keyActive,columnValues,keyPointer,valuePointer,
      sequenceLength,metadataIndex,rowStart,blockCount,columnCount,keyValueBase,backend,bop,ofFloat,toFloat]
  all_goals first | rfl | skip
  all_goals simp only [column_key_function,column_value_function,columnCount,metadataIndex,
    keyValueBase,Region.cast]
  all_goals rfl

theorem loop_range_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (ph : Bool) (first last stride : Nat) (a : Accumulator α c.BM c.BD)
    (h : Ready M c s t a) (hBN : 0<c.BN) :
    ∃ final, FP.Structural.range (backend M D) (indexName ph) first last stride (loopBody c ph) t = some final ∧
      Ready M c s final (foldRange (update M D c s ph) first last stride a) := by
  apply range_values (backend M D) (indexName ph) (loopBody c ph) (update M D c s ph)
    (fun _ a u => Ready M c s u a) (fun a u => Ready M c s u a) first last stride a t
  · intro n a u hu _
    exact loop_body_run M D c s u ph n a hu hBN
  · intro n a u hu
    exact hu
  · exact h

#axiomsClean column_key_function
#axiomsClean column_value_function
#axiomsClean loop_body_run
#axiomsClean loop_range_run
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
