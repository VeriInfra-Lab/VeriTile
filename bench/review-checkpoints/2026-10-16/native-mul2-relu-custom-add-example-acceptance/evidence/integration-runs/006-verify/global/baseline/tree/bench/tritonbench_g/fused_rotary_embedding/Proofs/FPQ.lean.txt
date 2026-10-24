import bench.tritonbench_g.fused_rotary_embedding.FusedRotaryEmbedding
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPQ
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (scatter_memory)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
structure Parameters where
  (q cosine sine : RegionName)
  (tokenStride headStride dimensionStride cosineTokenStride cosineStride half : Nat)

def qTyped (dtype : FloatDType)
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, Cos = DTYPE, Sin = DTYPE
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0)
  loaded_q1 = tl.load(Q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin)
  loaded_sin = tl.load(Sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0)
  tl.store(Q + off_q1, out_q1)
}

def source (p : Parameters) : ComputeKernel :=
  qTyped .fp32 p.q p.cosine p.sine p.tokenStride p.headStride p.dimensionStride
    p.cosineTokenStride p.cosineStride (p.half * 2) p.half

def legacySource (p : Parameters) : ComputeKernel :=
  decoding_fused_rotary_embedding_q_surface p.q p.cosine p.sine p.tokenStride p.headStride
    p.dimensionStride p.cosineTokenStride p.cosineStride (p.half * 2) p.half

theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source, legacySource, qTyped, decoding_fused_rotary_embedding_q_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure, VeriTile.Triton.eraseDType,
    FloatDType.numericDType, NumericDType.eraseDType]

def offset {α : Type} (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.half]) : Nat :=
  s.pids 1 * p.tokenStride + s.pids 0 * p.headStride +
    (i.1.val + if second then p.half else 0) * p.dimensionStride

def cacheOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.half]) : Nat :=
  s.pids 1 * p.cosineTokenStride + i.1.val * p.cosineStride

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.half]) : α :=
  let x0 := (s.mem p.q (offset p s Bool.false i)).read .fp32
  let x1 := (s.mem p.q (offset p s Bool.true i)).read .fp32
  let c := (s.mem p.cosine (cacheOffset p s i)).read .fp32
  let z := (s.mem p.sine (cacheOffset p s i)).read .fp32
  if second then M.binary none .fp32 .add
      (M.binary none .fp32 .mul x0 z) (M.binary none .fp32 .mul x1 c)
  else M.binary none .fp32 .sub
      (M.binary none .fp32 .mul x0 c) (M.binary none .fp32 .mul x1 z)

def halfScatter {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (second : Bool) (initial : State α) : State α :=
  (TileShape.allIndices [p.half]).foldl (fun t i =>
    t.write p.q (offset p s second i) (.mk .fp32 (value M p s second i))) initial

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  halfScatter M p s Bool.true (halfScatter M p s Bool.false s)

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, qTyped, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  simp [execution, written, halfScatter, scatter_memory, value, offset, cacheOffset,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPQ
