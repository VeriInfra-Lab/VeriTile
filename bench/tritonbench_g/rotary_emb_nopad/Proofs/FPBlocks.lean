import bench.tritonbench_g.rotary_emb_nopad.RotaryEmbNopad
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPBlocks
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
structure Parameters where
  (input cosine sine : RegionName)
  (tokenStride headStride dimensionStride cosineTokenStride cosineStride
    totalTokens heads groups half tokens : Nat)

def qTyped (dtype : FloatDType)
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, Cos = DTYPE, Sin = DTYPE
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  dim_range1 = dim_range0 + $(HEAD_HALF)
  off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
    dim_range0[None, :] * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  off_q0 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range0[None, None, :] * $(head_dim_stride)
  off_q1 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range1[None, None, :] * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)),
    other=0.0)
  loaded_q1 = tl.load(Q + off_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
  out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]
  tl.store(Q + off_q0, out_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)))
  tl.store(Q + off_q1, out_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)))
}

def kTyped (dtype : FloatDType)
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  tl.region K = DTYPE, Cos = DTYPE, Sin = DTYPE
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  handle_kv = (cur_head_idx % $(KV_GROUP_NUM)) == 0
  if handle_kv {
    k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
    dim_range0 = tl.arange(0, $(HEAD_HALF))
    dim_range1 = dim_range0 + $(HEAD_HALF)
    off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
      dim_range0[None, :] * $(cos_stride)
    loaded_cos = tl.load(Cos + off_cos_sin,
      mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
    loaded_sin = tl.load(Sin + off_cos_sin,
      mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
    off_k0 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range0[None, None, :] * $(head_dim_stride)
    off_k1 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range1[None, None, :] * $(head_dim_stride)
    loaded_k0 = tl.load(K + off_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens), other=0.0)
    loaded_k1 = tl.load(K + off_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens), other=0.0)
    out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
    tl.store(K + off_k0, out_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
    tl.store(K + off_k1, out_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
  }
}

def source (key : Bool) (p : Parameters) : ComputeKernel :=
  if key then kTyped .fp32 p.input p.cosine p.sine p.tokenStride p.headStride p.dimensionStride
    p.cosineTokenStride p.cosineStride p.totalTokens p.groups p.half p.tokens
  else qTyped .fp32 p.input p.cosine p.sine p.tokenStride p.headStride p.dimensionStride
    p.cosineTokenStride p.cosineStride p.totalTokens p.heads p.half p.tokens

def legacySource (key : Bool) (p : Parameters) : ComputeKernel :=
  if key then rotary_embedding_k_surface p.input p.cosine p.sine p.tokenStride p.headStride
    p.dimensionStride p.cosineTokenStride p.cosineStride p.totalTokens p.groups p.half p.tokens
  else rotary_embedding_q_surface p.input p.cosine p.sine p.tokenStride p.headStride
    p.dimensionStride p.cosineTokenStride p.cosineStride p.totalTokens p.heads p.half p.tokens

theorem source_projection (key : Bool) (p : Parameters) :
    (source key p).eraseDType.surfaceBody = (legacySource key p).eraseDType.surfaceBody := by
  cases key <;> simp [source, legacySource, qTyped, kTyped,
    rotary_embedding_q_surface, rotary_embedding_k_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, VeriTile.Triton.eraseDType]
  all_goals conv_rhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, VeriTile.Triton.eraseDType]

abbrev Index (p : Parameters) := TileIndex [p.tokens, 1, p.half]

def token {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 1 * p.tokens + i.1.val

def head {α : Type} (key : Bool) (p : Parameters) (s : State α) : Nat :=
  if key then s.pids 0 / p.groups else s.pids 0

def offset {α : Type} (key : Bool) (p : Parameters) (s : State α)
    (second : Bool) (i : Index p) : Nat :=
  token p s i * p.tokenStride + head key p s * p.headStride +
    (i.2.2.1.val + if second then p.half else 0) * p.dimensionStride

def cacheOffset {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  token p s i * p.cosineTokenStride + i.2.2.1.val * p.cosineStride

def active {α : Type} (key : Bool) (p : Parameters) (s : State α) (i : Index p) : Prop :=
  (if key then s.pids 0 % p.groups = 0 else s.pids 0 < p.heads) ∧ token p s i < p.totalTokens
instance {α : Type} (key : Bool) (p : Parameters) (s : State α) :
    DecidablePred (active key p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def value {α : Type} [Inhabited α] (M : Algebra α) (key : Bool) (p : Parameters) (s : State α)
    (second : Bool) (i : Index p) : α :=
  let x0 := (s.mem p.input (offset key p s Bool.false i)).read .fp32
  let x1 := (s.mem p.input (offset key p s Bool.true i)).read .fp32
  let c := (s.mem p.cosine (cacheOffset p s i)).read .fp32
  let z := (s.mem p.sine (cacheOffset p s i)).read .fp32
  if second then M.binary none .fp32 .add
      (M.binary none .fp32 .mul x0 z) (M.binary none .fp32 .mul x1 c)
  else M.binary none .fp32 .sub
      (M.binary none .fp32 .mul x0 c) (M.binary none .fp32 .mul x1 z)

def halfScatter {α : Type} [Inhabited α] (M : Algebra α) (key : Bool) (p : Parameters)
    (s : State α) (second : Bool) (initial : State α) : State α :=
  (TileShape.allIndices [p.tokens, 1, p.half]).foldl (fun t i => if active key p s i then
    t.write p.input (offset key p s second i) (.mk .fp32 (value M key p s second i)) else t) initial

def written {α : Type} [Inhabited α] (M : Algebra α) (key : Bool) (p : Parameters) (s : State α) : State α :=
  halfScatter M key p s Bool.true (halfScatter M key p s Bool.false s)

def execution {α : Type} [Inhabited α] (M : Algebra α) (key : Bool) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source key p) s = some t} := by
  cases key <;> by_cases h : s.pids 0 % p.groups = 0 <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, qTyped, kTyped, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
       Region.cast, Option.bind, integralFloorDiv, integralMod, h]
     try rfl)

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (key : Bool) (p : Parameters) (s : State α) :
    (execution M key p s).val.mem = (written M key p s).mem := by
  cases key <;> by_cases h : s.pids 0 % p.groups = 0
  all_goals simp +contextual [execution, written, halfScatter, masked_scatter_memory,
    value, offset, cacheOffset, token, head, active,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, h,
    integralFloorDiv, integralMod, and_assoc, and_left_comm, and_comm,
    TileShape.insertAxis, TileShape.dropInsertedIndex,
    TileShape.dropInsertedIndex_two_pair, TileShape.dropInsertedIndex_zero_pair]
  all_goals rfl

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPBlocks
