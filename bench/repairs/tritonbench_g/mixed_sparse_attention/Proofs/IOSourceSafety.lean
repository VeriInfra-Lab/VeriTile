import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Whole
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MixedSparseAttention.IOSourceSafety
open VeriTile Triton FP.Structural
open Whole002 Source
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Source-free addresses of the actual masked load/store families. The public
array IO derives these finite bounds; this is not a numerical assumption. -/
structure Bounds (c : Config) (s : State α) (capacity : RegionName → Nat) : Prop where
  sequence : s.pids 1/c.H < capacity (Region.cast c.seq)
  blocks : metadataIndex c s < capacity (Region.cast c.blocks)
  counts : metadataIndex c s < capacity (Region.cast c.counts)
  offsets : ∀ n, n<blockCount c s → metadataIndex c s*c.NS+n < capacity (Region.cast c.offsets)
  columns : ∀ n (j : Fin c.BN), keyActive c s Bool.true n j →
    metadataIndex c s*c.NV+n+j.val < capacity (Region.cast c.columns)
  query : ∀ i : TileIndex [c.BM,c.BD], rowStart c s+i.1.val<sequenceLength c s →
    queryBase c s+(rowStart c s+i.1.val)*c.sqm+i.2.1.val*c.sqk < capacity c.q
  key : ∀ ph n (j : Fin c.BN) (d : Fin c.BD), keyActive c s ph n j →
    keyValueBase c s+d.val*c.skk+columnValues c s ph n j*c.skn < capacity c.k
  value : ∀ ph n (j : Fin c.BN) (d : Fin c.BD), keyActive c s ph n j →
    keyValueBase c s+d.val*c.svk+columnValues c s ph n j*c.svn < capacity c.v
  output : ∀ i : TileIndex [c.BM,c.BD], rowStart c s+i.1.val<sequenceLength c s →
    (outputPointer c s i).2 < capacity c.out

theorem prelude_safe (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    Safety.traceSafeList (backend M D) capacity (prelude c) s := by
  cases hs : c.storage
  all_goals
    simp (config := {maxSteps := 1200000}) [prelude,activeBody,parsedActiveBody,parsedKernel,hs,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.Storage.tile,TypedSource.Storage.float,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,Op.tritonDiv,Op.tritonMod,
      backend,Region.cast,ofFloat,toFloat,State.setReg,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    refine ⟨hb.sequence,hb.blocks,hb.counts,?_⟩
    intro i d hi
    exact hb.query (i,d,()) hi

theorem body_safe (M : Algebra α) (D : Dot α) (c : Config) (s t : State α)
    (ph : Bool) (n : Nat) (a : Accumulator α c.BM c.BD) (h : Ready M c s t a)
    (hbn : 0<c.BN) (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    Safety.traceSafeList (backend M D) capacity (loopBody c ph)
      (t.setReg (indexName ph) .nat [] (fun _=>n)) := by
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage <;> cases ph
  all_goals
    have hq:=h.query
    simp only [TypedSource.Storage.tile,TypedSource.Storage.float,FloatDType.toTileDType,ofFloat] at hq
    simp (config := {maxSteps := 2000000}) [loopBody,indexName,columnBody,blockBody,activeBody,
      parsedActiveBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,
      TypedSource.Storage.tile,TypedSource.Storage.float,maximumAssignment,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,natLe,numericLt,numericLe,numericEq,integralFloorDiv,integralMod,backend,
      Region.cast,ofFloat,toFloat,State.setReg,h.seqlen,h.rows,h.lanes,h.keyptr,h.valueptr,h.outptr,
      h.numblocks,h.blockptr,h.numcolumns,h.columnptr,hq,h.mask,h.maximum,h.normalizer,h.output,
      h.maxblocks,h.mem,h.pids,h.undef,TileShape.axisDim,hbn,and_assoc,
      keyPointer,valuePointer,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    first
    | refine ⟨fun hn=>hb.offsets n hn,?_,?_⟩
    | refine ⟨fun j hj hn=>hb.columns n j ⟨hj,hn⟩,?_,?_⟩
  all_goals
    first
    | intro d j hj hn
      simpa [keyActive,columnValues,hj,hn,Nat.add_assoc] using hb.key Bool.false n j d ⟨hj,hn⟩
    | intro d j hj hn
      simpa [keyActive,columnValues,hj,hn,Nat.add_assoc] using hb.key Bool.true n j d ⟨hj,hn⟩
    | intro j d hj hn
      simpa [keyActive,columnValues,hj,hn,Nat.add_assoc] using hb.value Bool.false n j d ⟨hj,hn⟩
    | intro j d hj hn
      simpa [keyActive,columnValues,hj,hn,Nat.add_assoc] using hb.value Bool.true n j d ⟨hj,hn⟩
    | split <;> trivial

theorem between_safe (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (capacity : RegionName → Nat) :
    Safety.traceSafeList (backend M D) capacity (between c) s := by
  cases hs : c.storage <;>
    simp [between,activeBody,parsedActiveBody,parsedKernel,hs,TypedSource.FP16.source,
      TypedSource.BF16.source,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  all_goals split <;> trivial

theorem suffix_safe (M : Algebra α) (D : Dot α) (c : Config) (s t : State α)
    (a : Accumulator α c.BM c.BD) (h : Ready M c s t a)
    (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    Safety.traceSafeList (backend M D) capacity (suffix c) t := by
  cases hs : c.storage
  all_goals
    simp (config := {maxSteps := 900000}) [suffix,activeBody,parsedActiveBody,parsedKernel,hs,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.Storage.tile,TypedSource.Storage.float,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,numericEq,backend,Region.cast,ofFloat,toFloat,
      h.output,h.normalizer,h.outptr,h.mask,State.setReg,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    constructor
    · intro i d tail hi
      exact hb.output (i,d,tail) hi
    · split <;> trivial
end
end VeriTile.Bench.Repairs.MixedSparseAttention.IOSourceSafety
