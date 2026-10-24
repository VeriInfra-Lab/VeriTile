import bench.tritonbench_g.rope_embedding.RopeEmbedding
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
structure Parameters where
  (input cosine sine : RegionName)
  (rowStride cosineStride sineStride length dimension heads groupSize width : Nat)

def forwardFirst (dtype : FloatDType)
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, cos = DTYPE, sin = DTYPE
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}

def forwardSecond (dtype : FloatDType)
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, cos = DTYPE, sin = DTYPE
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}

def backwardFirst (dtype : FloatDType)
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, cos = DTYPE, sin = DTYPE
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}

def backwardSecond (dtype : FloatDType)
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, cos = DTYPE, sin = DTYPE
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}

def source (backward second : Bool) (p : Parameters) : ComputeKernel :=
  let stage := if backward then (if second then backwardSecond else backwardFirst)
    else (if second then forwardSecond else forwardFirst)
  stage .fp32 p.input p.cosine p.sine p.rowStride p.cosineStride p.sineStride p.length
    p.dimension p.heads p.groupSize p.width

def legacySource (backward second : Bool) (p : Parameters) : ComputeKernel :=
  let stage := if backward then (if second then rope_embedding_backward_second_half else rope_embedding_backward_first_half)
    else (if second then rope_embedding_forward_second_half else rope_embedding_forward_first_half)
  stage p.input p.cosine p.sine p.rowStride p.cosineStride p.sineStride p.length
    p.dimension p.heads p.groupSize p.width

def offset {α : Type} (p : Parameters) (s : State α) (second : Bool) (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.rowStride + s.pids 1 * p.groupSize * p.dimension + i.1.val +
    if second then p.dimension / 2 else 0

def coefficientOffset {α : Type} (p : Parameters) (s : State α) (stride : Nat)
    (i : TileIndex [p.width]) : Nat := s.pids 0 % p.length * stride + i.1.val

def active {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Prop :=
  i.1.val < p.dimension / 2 ∧ s.pids 1 * p.groupSize < p.heads
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- Exact operation order, including the actual sign-bit negation before the
backward products and both load-to-coefficient register casts. -/
def value {α : Type} [Inhabited α] (M : Algebra α) (backward second : Bool)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) : α :=
  let x0 := M.cast none .fp32 .fp32 ((s.mem p.input (offset p s Bool.false i)).read .fp32)
  let x1 := M.cast none .fp32 .fp32 ((s.mem p.input (offset p s Bool.true i)).read .fp32)
  let c := (s.mem p.cosine (coefficientOffset p s p.cosineStride i)).read .fp32
  let rawSine := (s.mem p.sine (coefficientOffset p s p.sineStride i)).read .fp32
  let sn := if backward then M.unary none (.neg .fp32) rawSine else rawSine
  if second then M.binary none .fp32 .add
      (M.binary none .fp32 .mul x1 c) (M.binary none .fp32 .mul x0 sn)
  else M.binary none .fp32 .sub
      (M.binary none .fp32 .mul x0 c) (M.binary none .fp32 .mul x1 sn)

def written {α : Type} [Inhabited α] (M : Algebra α) (backward second : Bool)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i => if active p s i then
    t.write p.input (offset p s second i) (.mk .fp32 (value M backward second p s i)) else t) s

def execution {α : Type} [Inhabited α] (M : Algebra α) (backward second : Bool)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source backward second p) s = some t} := by
  cases backward <;> cases second <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, forwardFirst, forwardSecond, backwardFirst, backwardSecond,
       FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
       FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
       Region.cast, Option.bind, integralFloorDiv, integralMod]
     try rfl)

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (backward second : Bool)
    (p : Parameters) (s : State α) :
    (execution M backward second p s).val.mem = (written M backward second p s).mem := by
  cases backward <;> cases second
  all_goals simp +contextual [execution, written, masked_scatter_memory, value, offset,
    coefficientOffset, active, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Region.cast, integralFloorDiv, integralMod, and_assoc, and_left_comm, and_comm, Nat.add_assoc]

#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
