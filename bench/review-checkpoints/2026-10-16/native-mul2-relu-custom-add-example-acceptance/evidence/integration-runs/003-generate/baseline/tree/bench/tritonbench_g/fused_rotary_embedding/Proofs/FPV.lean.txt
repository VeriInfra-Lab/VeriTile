import bench.tritonbench_g.fused_rotary_embedding.FusedRotaryEmbedding
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPV
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (copyBackend scatter_memory)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
structure Parameters where
  (input output : RegionName)
  (blockTables lengths : Region .nat)
  (groups tokenStride headStride dimensionStride cacheBlockStride cacheHeadStride
    cacheSlotStride cacheDimensionStride tableSequenceStride tableBlockStride blockSize width : Nat)

def vTyped (dtype : FloatDType)
    (v v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    ComputeKernel := triton {
  tl.region v = DTYPE, v_cache = DTYPE
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range = tl.arange(0, $(HEAD_DIM))
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    loaded_v = tl.load(v + off_kv + dim_range * $(head_dim_stride))
    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    v_range = block_ids * $(vcb_stride) + cur_k_head_idx * $(vch_stride) +
      offsets_in_last_block * $(vcs_stride) + dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}

def source (p : Parameters) : ComputeKernel :=
  vTyped .fp32 p.input p.output p.blockTables p.lengths p.groups p.tokenStride p.headStride
    p.dimensionStride p.cacheBlockStride p.cacheHeadStride p.cacheSlotStride p.cacheDimensionStride
    p.tableSequenceStride p.tableBlockStride p.blockSize p.width

def legacySource (p : Parameters) : ComputeKernel :=
  decoding_fused_rotary_embedding_vcache_chain p.input p.output p.blockTables p.lengths p.groups
    p.tokenStride p.headStride p.dimensionStride p.cacheBlockStride p.cacheHeadStride
    p.cacheSlotStride p.cacheDimensionStride p.tableSequenceStride p.tableBlockStride p.blockSize p.width

theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source, legacySource, vTyped, decoding_fused_rotary_embedding_vcache_chain,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure, VeriTile.Triton.eraseDType,
    FloatDType.numericDType, NumericDType.eraseDType]

def active {α : Type} (p : Parameters) (s : State α) : Prop := s.pids 0 % p.groups = 0
instance {α : Type} (p : Parameters) (s : State α) : Decidable (active p s) :=
  inferInstanceAs (Decidable (_ = _))

def past {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.lengths.cast (s.pids 1)).read .nat - 1

def blockId {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.blockTables.cast
    (s.pids 1 * p.tableSequenceStride + past p s / p.blockSize * p.tableBlockStride)).read .nat

def inputOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  s.pids 1 * p.tokenStride + s.pids 0 / p.groups * p.headStride + i.1.val * p.dimensionStride

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  blockId p s * p.cacheBlockStride + s.pids 0 / p.groups * p.cacheHeadStride +
    past p s % p.blockSize * p.cacheSlotStride + i.1.val * p.cacheDimensionStride

def copied {α : Type} [Inhabited α] (p : Parameters) (s : State α) : State α :=
  if active p s then (TileShape.allIndices [p.width]).foldl (fun t i =>
    t.write p.output (outputOffset p s i)
      (.mk .fp32 ((s.mem p.input (inputOffset p s i)).read .fp32))) s else s

def execution {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (source p) s = some t} := by
  by_cases h : s.pids 0 % p.groups = 0 <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, vTyped, copyBackend, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
       integralFloorDiv, integralMod, FloatDType.numericDType, FloatDType.toTileDType, h]
     try rfl)

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) (s : State α) :
    (execution rest p s).val.mem = (copied p s).mem := by
  by_cases h : s.pids 0 % p.groups = 0 <;>
    simp [execution, copied, scatter_memory, inputOffset, outputOffset, blockId, past, active,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, h,
      integralFloorDiv, integralMod]

theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) (s : State α)
    (ha : active p s) (hinj : Function.Injective (outputOffset p s)) (i : TileIndex [p.width]) :
    (execution rest p s).val.mem p.output (outputOffset p s i) =
      .mk .fp32 ((s.mem p.input (inputOffset p s i)).read .fp32) := by
  rw [execution_memory]
  simp only [copied, ha, ↓reduceIte]
  exact State.scatter_readback _ _ _ _ i hinj

def Outside {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p s → o ≠ outputOffset p s i

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (execution rest p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  by_cases ha : active p s
  · simp only [copied, ha, ↓reduceIte]
    exact State.scatter_frame _ _ _ _ _ _ (h.imp id (fun h i _ => h i ha)) s
  · simp [copied, ha]

def Post {α : Type} [Inhabited α] (p : Parameters) (initial final : State α) : Prop :=
  (∀ i, active p initial → final.mem p.output (outputOffset p initial i) =
    .mk .fp32 ((initial.mem p.input (inputOffset p initial i)).read .fp32)) ∧
  (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o)

/-- The injectivity gate is the same active-write constraint carried by the
original ChainMetaGroupedMasked2DKernelIO contract, not a floating-point law. -/
theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) (s : State α)
    (hinj : active p s → Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (copyBackend rest) (source p) s = some final ∧ Post p s final :=
  ⟨(execution rest p s).val, (execution rest p s).property,
    fun i ha => readback rest p s ha (hinj ha) i, frame rest p s⟩

theorem output_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (hstride : p.cacheDimensionStride ≠ 0) : Function.Injective (outputOffset p s) := by
  intro i j h
  have hi := (Nat.mul_left_inj hstride).mp (Nat.add_left_cancel h)
  exact Prod.ext (Fin.ext hi) (Subsingleton.elim _ _)

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
#axiomsClean correctness
#axiomsClean output_injective
end
end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPV
