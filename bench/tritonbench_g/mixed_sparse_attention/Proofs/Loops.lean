import bench.tritonbench_g.mixed_sparse_attention.Proofs.Prefix
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 15000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def loopBody (c : Config) (columnPhase : Bool) : List ComputeStmt :=
  if columnPhase then columnBody c else blockBody c

def indexName (columnPhase : Bool) : RegName := if columnPhase then "start_n" else "block_index"

theorem tile1_eta {m : Nat} (i : TileIndex [m]) : (i.1,())=i := by
  rcases i with ⟨j,⟨⟩⟩
  rfl

theorem tile2_eta {m n : Nat} (i : TileIndex [m,n]) : (i.1,i.2.1,())=i := by
  rcases i with ⟨j,k,⟨⟩⟩
  rfl

theorem loop_body_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (ph : Bool) (start : Nat) (a : Accumulator α c.BM c.BD)
    (h : Ready M c s t a) (hBN : 0<c.BN) :
    ∃ final, run (backend M D) (loopBody c ph)
      (t.setReg (indexName ph) .nat [] (fun _ => start)) = some final ∧
      Ready M c s final (update M D c s ph start a) := by
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage <;> cases ph
  all_goals
    have hq := h.query
    simp only [TypedSource.Storage.tile,TypedSource.Storage.float,FloatDType.toTileDType,ofFloat] at hq
    simp [loopBody,indexName,columnBody,blockBody,activeBody,parsedActiveBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.Storage.tile,TypedSource.Storage.float,
      maximumAssignment,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,numericEq,
      integralFloorDiv,integralMod,backend,Region.cast,ofFloat,toFloat,State.setReg,
      h.seqlen,h.rows,h.lanes,h.keyptr,h.valueptr,h.outptr,h.numblocks,h.blockptr,
      h.numcolumns,h.columnptr,hq,h.mask,h.maximum,h.normalizer,h.output,h.maxblocks,
      h.mem,h.pids,h.undef,TileShape.axisDim,hBN, and_assoc]
  all_goals constructor <;>
    try simp [TypedSource.Storage.tile,TypedSource.Storage.float,State.setReg,h.seqlen,h.rows,h.lanes,h.keyptr,h.valueptr,h.outptr,h.numblocks,h.blockptr,
      h.numcolumns,h.columnptr,hq,h.mask,h.maximum,h.normalizer,h.output,h.maxblocks,
      h.mem,h.pids,h.undef,and_assoc,update,keyValues,valueValues,keyActive,columnValues,keyPointer,valuePointer,
      sequenceLength,metadataIndex,rowStart,blockCount,columnCount,keyValueBase,backend,bop,ofFloat,toFloat]
  all_goals try delta keyValues valueValues bop
  all_goals dsimp only [bop,keyValues,valueValues,keyActive,columnValues,
    keyValueBase,sequenceLength,blockCount,columnCount,metadataIndex,rowStart,
    TypedSource.Storage.tile,TypedSource.Storage.float,ofFloat,toFloat,
    Region.cast,Function.comp_apply]
  all_goals simp only [and_assoc,Nat.add_assoc,show (0.0 : ℝ)=0 by norm_num]
  all_goals try simp only [tile1_eta,tile2_eta]
  all_goals first | rfl | (funext i; simp [bop,Function.comp_apply,tile1_eta,tile2_eta,Broadcast.leftIndex,Broadcast.rightIndex,FloatDType.toTileDType])

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

#axiomsClean loop_body_run
#axiomsClean loop_range_run
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
