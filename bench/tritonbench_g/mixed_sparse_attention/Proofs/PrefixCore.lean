import bench.tritonbench_g.mixed_sparse_attention.Proofs.MixedSparseAttention_Source_Reference
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def keyPointer {α : Type} (c : Config) (s : State α) : TileIndex [c.BD,1] → RegionName×Nat :=
  fun i => (c.k,keyValueBase c s+i.1.val*c.skk)
def valuePointer {α : Type} (c : Config) (s : State α) : TileIndex [1,c.BD] → RegionName×Nat :=
  fun i => (c.v,keyValueBase c s+i.2.1.val*c.svk)
def outputPointer {α : Type} (c : Config) (s : State α) : TileIndex [c.BM,c.BD] → RegionName×Nat :=
  fun i => (c.out,queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok)

structure Ready {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (s t : State α)
    (a : Accumulator α c.BM c.BD) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  seqlen : t.regs .nat [] "seqlen" = some (fun _ => sequenceLength c s)
  rows : t.regs .nat [c.BM] "offs_m" = some (fun i => rowStart c s+i.1.val)
  lanes : t.regs .nat [c.BN] "offs_n" = some (fun i => i.1.val)
  keyptr : t.regs .ptr [c.BD,1] "k_ptrs" = some (keyPointer c s)
  valueptr : t.regs .ptr [1,c.BD] "v_ptrs" = some (valuePointer c s)
  outptr : t.regs .ptr [c.BM,c.BD] "o_ptrs" = some (outputPointer c s)
  numblocks : t.regs .nat [] "num_blks" = some (fun _ => blockCount c s)
  blockptr : t.regs .ptr [] "blks_ptr" = some (fun _ => (Region.cast c.offsets,metadataIndex c s*c.NS))
  numcolumns : t.regs .nat [] "num_cols" = some (fun _ => columnCount c s)
  columnptr : t.regs .ptr [] "cols_ptr" = some (fun _ => (Region.cast c.columns,metadataIndex c s*c.NV))
  query : t.regs c.storage.tile [c.BM,c.BD] "q" = some (fun i=>ofFloat c.storage.float (queryValues M c s i))
  mask : t.regs .bool [c.BM,1] "m_mask" = some (fun i => decide (rowStart c s+i.1.val < sequenceLength c s))
  maximum : t.regs .real [c.BM] "m_i" = some a.maximum
  normalizer : t.regs .real [c.BM] "l_i" = some a.normalizer
  output : t.regs .real [c.BM,c.BD] "acc" = some a.output
  maxblocks : t.regs .nat [] "max_num_blks" = some (fun _ => blockCount c s)

def prelude (c : Config) : List ComputeStmt := (parsedKernel c).surfaceBody.take 3 ++ (activeBody c).take 21

def prefixExec {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α) :
    { final : State α // run (backend M D) (prelude c) s = some final } := by
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [prelude,activeBody,parsedActiveBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,
      TypedSource.Storage.tile,TypedSource.Storage.float,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,Op.tritonDiv,Op.tritonMod,
      backend,Region.cast,ofFloat,toFloat,State.setReg]
    rfl


end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
