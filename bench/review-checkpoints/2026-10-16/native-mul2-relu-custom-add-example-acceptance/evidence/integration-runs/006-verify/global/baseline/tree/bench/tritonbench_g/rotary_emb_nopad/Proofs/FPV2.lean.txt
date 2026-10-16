import bench.tritonbench_g.rotary_emb_nopad.RotaryEmbNopad
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPV2
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
structure Parameters where
  (input output cosine sine : RegionName)
  (blockTables lengths : Region .nat)
  (tokenStride headStride dimensionStride cosineTokenStride cosineStride
    cacheBlockStride cacheHeadStride cacheSlotStride cacheDimensionStride
    tableSequenceStride tableBlockStride blockSize totalTokens heads half : Nat)

def qTyped (dtype : FloatDType)
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, Cos = DTYPE, Sin = DTYPE
  block_head_index = tl.program_id(0)
  block_token_index = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  dim_range1 = dim_range0 + $(HEAD_HALF)
  off_cos_sin = block_token_index * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  off_q0 = block_token_index * $(q_token_stride) +
    block_head_index * $(q_head_stride) + dim_range0 * $(head_dim_stride)
  off_q1 = block_token_index * $(q_token_stride) +
    block_head_index * $(q_head_stride) + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0,
    mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  loaded_q1 = tl.load(Q + off_q1,
    mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0, mask=block_head_index < $(Q_HEAD_NUM))
  tl.store(Q + off_q1, out_q1, mask=block_head_index < $(Q_HEAD_NUM))
}

def cacheTyped (dtype : FloatDType)
    (KVCache K Cos Sin : RegionName) (BlockTables ContextLengths : Region .nat)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
      block_size q_total_tokens Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  tl.region K = DTYPE, Cos = DTYPE, Sin = DTYPE, KVCache = DTYPE
  block_head_index = tl.program_id(0)
  block_token_index = tl.program_id(1)
  dim = tl.arange(0, $(HEAD_HALF))
  dim1 = dim + $(HEAD_HALF)
  off_cos_sin = block_token_index * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  off_k0 = block_token_index * $(k_token_stride) +
    block_head_index * $(k_head_stride) + dim * $(head_dim_stride)
  off_k1 = block_token_index * $(k_token_stride) +
    block_head_index * $(k_head_stride) + dim1 * $(head_dim_stride)
  loaded_k0 = tl.load(K + off_k0, mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  loaded_k1 = tl.load(K + off_k1, mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
  out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
  past_kv_seq_len = tl.load(ContextLengths + block_token_index) - $(1)
  last_block_idx = past_kv_seq_len // $(block_size)
  block_table_ptr = BlockTables + block_token_index * $(bts_stride)
  block_ids = tl.load(block_table_ptr + last_block_idx * $(btb_stride),
    mask=block_token_index < $(q_total_tokens))
  offsets_in_last_block = (past_kv_seq_len % $(block_size)) * $(cachebs_stride)
  kv_range0 = block_ids * $(cacheb_stride) +
    block_head_index * $(cacheh_stride) + offsets_in_last_block +
    dim * $(cached_stride)
  kv_range1 = block_ids * $(cacheb_stride) +
    block_head_index * $(cacheh_stride) + offsets_in_last_block +
    dim1 * $(cached_stride)
  tl.store(KVCache + kv_range0, out_k0, mask=block_head_index < $(Q_HEAD_NUM))
  tl.store(KVCache + kv_range1, out_k1, mask=block_head_index < $(Q_HEAD_NUM))
}

def source (cache : Bool) (p : Parameters) : ComputeKernel :=
  if cache then cacheTyped .fp32 p.output p.input p.cosine p.sine p.blockTables p.lengths
    p.tokenStride p.headStride p.dimensionStride p.cosineTokenStride p.cosineStride
    p.cacheBlockStride p.cacheHeadStride p.cacheSlotStride p.cacheDimensionStride
    p.tableSequenceStride p.tableBlockStride p.blockSize p.totalTokens p.heads p.half
  else qTyped .fp32 p.input p.cosine p.sine p.tokenStride p.headStride p.dimensionStride
    p.cosineTokenStride p.cosineStride p.totalTokens p.heads p.half

def legacySource (cache : Bool) (p : Parameters) : ComputeKernel :=
  if cache then fused_rotary_embedding_v2_cache_surface p.output p.input p.cosine p.sine p.blockTables p.lengths
    p.tokenStride p.headStride p.dimensionStride p.cosineTokenStride p.cosineStride
    p.cacheBlockStride p.cacheHeadStride p.cacheSlotStride p.cacheDimensionStride
    p.tableSequenceStride p.tableBlockStride p.blockSize p.totalTokens p.heads p.half
  else fused_rotary_embedding_v2_q_surface p.input p.cosine p.sine p.tokenStride p.headStride p.dimensionStride
    p.cosineTokenStride p.cosineStride p.totalTokens p.heads p.half

theorem source_projection (cache : Bool) (p : Parameters) :
    (source cache p).eraseDType.surfaceBody = (legacySource cache p).eraseDType.surfaceBody := by
  cases cache <;> simp [source, legacySource, qTyped, cacheTyped,
    fused_rotary_embedding_v2_q_surface, fused_rotary_embedding_v2_cache_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, VeriTile.Triton.eraseDType]

def inputOffset {α : Type} (p : Parameters) (s : State α) (second : Bool) (i : TileIndex [p.half]) : Nat :=
  s.pids 1 * p.tokenStride + s.pids 0 * p.headStride +
    (i.1.val + if second then p.half else 0) * p.dimensionStride

def cacheOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.half]) : Nat :=
  s.pids 1 * p.cosineTokenStride + i.1.val * p.cosineStride

def past {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.lengths.cast (s.pids 1)).read .nat - 1

def blockId {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  if s.pids 1 < p.totalTokens then
    (s.mem p.blockTables.cast
      (s.pids 1 * p.tableSequenceStride + past p s / p.blockSize * p.tableBlockStride)).read .nat
  else s.undef .nat p.blockTables.cast
    (s.pids 1 * p.tableSequenceStride + past p s / p.blockSize * p.tableBlockStride)

def outputRegion (cache : Bool) (p : Parameters) : RegionName := if cache then p.output else p.input

def outputOffset {α : Type} [Inhabited α] (cache : Bool) (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.half]) : Nat :=
  if cache then blockId p s * p.cacheBlockStride + s.pids 0 * p.cacheHeadStride +
    past p s % p.blockSize * p.cacheSlotStride +
    (i.1.val + if second then p.half else 0) * p.cacheDimensionStride
  else inputOffset p s second i

def active {α : Type} (p : Parameters) (s : State α) (_ : TileIndex [p.half]) : Prop := s.pids 0 < p.heads
instance {α : Type} (p : Parameters) (s : State α) :
    DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ < _))

def zero {α : Type} (M : Algebra α) : α := M.cast none .real .fp32 (M.literal none .real 0.0)

def coefficient {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (i : TileIndex [p.half]) : α :=
  if s.pids 1 < p.totalTokens then (s.mem r (cacheOffset p s i)).read .fp32 else zero M

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.half]) : α :=
  let x0 := (s.mem p.input (inputOffset p s Bool.false i)).read .fp32
  let x1 := (s.mem p.input (inputOffset p s Bool.true i)).read .fp32
  let c := coefficient M p s p.cosine i
  let z := coefficient M p s p.sine i
  if second then M.binary none .fp32 .add
      (M.binary none .fp32 .mul x0 z) (M.binary none .fp32 .mul x1 c)
  else M.binary none .fp32 .sub
      (M.binary none .fp32 .mul x0 c) (M.binary none .fp32 .mul x1 z)

def halfScatter {α : Type} [Inhabited α] (M : Algebra α) (cache : Bool) (p : Parameters)
    (s : State α) (second : Bool) (initial : State α) : State α :=
  (TileShape.allIndices [p.half]).foldl (fun t i => if active p s i then
    t.write (outputRegion cache p) (outputOffset cache p s second i)
      (.mk .fp32 (value M p s second i)) else t) initial

def written {α : Type} [Inhabited α] (M : Algebra α) (cache : Bool) (p : Parameters) (s : State α) : State α :=
  halfScatter M cache p s Bool.true (halfScatter M cache p s Bool.false s)

def execution {α : Type} [Inhabited α] (M : Algebra α) (cache : Bool) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source cache p) s = some t} := by
  cases cache <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, qTyped, cacheTyped, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
       Region.cast, Option.bind, integralFloorDiv, integralMod]
     try rfl)

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (cache : Bool) (p : Parameters) (s : State α) :
    (execution M cache p s).val.mem = (written M cache p s).mem := by
  cases cache
  all_goals simp +contextual [execution, written, halfScatter, masked_scatter_memory,
    value, coefficient, zero, inputOffset, cacheOffset, outputOffset, outputRegion, past, blockId, active,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    integralFloorDiv, integralMod]

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPV2
