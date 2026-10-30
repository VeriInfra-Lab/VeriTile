/-
Source-audit target: _rope_embedding
Source-audit Lean: rope
-/
import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Canonical `_rope_embedding` source, selected FP32 storage/compute profile.
Both directions, the four-head dynamic loop, both loaded halves and both
stores are retained. The explicit store conversion is the implicit pointer
conversion in Python. No arithmetic optimization is introduced. -/
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.Kernels
open VeriTile.Triton

def rope (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) : ComputeKernel :=
  triton {
    tl.ports inputs=[Q,cos,sin], outputs=[Q]
    tl.region Q = tl.float32, cos = tl.float32, sin = tl.float32
    ROPE_GROUP_SIZE = $(4)
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, $(BLOCK_SIZE))
    half_head_dim = $(head_dim / 2)
    mask = col_offsets < half_head_dim
    sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + half_head_dim * $(0) + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + half_head_dim * $(0) + col_offsets, mask=mask, other=0)
    if $((BACKWARD_PASS : Bool)) {
      sin1 = -sin1
    }
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, $(n_heads))
    for k in range(head_start, head_end, $(1)) {
      offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
      offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets + half_head_dim
      Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(tl.float32)
      Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(tl.float32)
      tl.store(Q + offs_q1, (Q1 * cos1 - Q2 * sin1).to(tl.float32), mask=mask)
      tl.store(Q + offs_q2, (Q2 * cos1 + Q1 * sin1).to(tl.float32), mask=mask)
    }
  }

/-- Existing head-group integer commutation candidate; every numerical operation is retained. -/
def ropeIndexCommuted (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) : ComputeKernel :=
  triton {
    tl.ports inputs=[Q,cos,sin], outputs=[Q]
    tl.region Q = tl.float32, cos = tl.float32, sin = tl.float32
    ROPE_GROUP_SIZE = $(4)
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, $(BLOCK_SIZE))
    half_head_dim = $(head_dim / 2)
    mask = col_offsets < half_head_dim
    sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + half_head_dim * $(0) + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + half_head_dim * $(0) + col_offsets, mask=mask, other=0)
    if $((BACKWARD_PASS : Bool)) {
      sin1 = -sin1
    }
    head_start = ROPE_GROUP_SIZE * group_head_position
    head_end = min(head_start + ROPE_GROUP_SIZE, $(n_heads))
    for k in range(head_start, head_end, $(1)) {
      offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
      offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets + half_head_dim
      Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(tl.float32)
      Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(tl.float32)
      tl.store(Q + offs_q1, (Q1 * cos1 - Q2 * sin1).to(tl.float32), mask=mask)
      tl.store(Q + offs_q2, (Q2 * cos1 + Q1 * sin1).to(tl.float32), mask=mask)
    }
  }

end VeriTile.Bench.Repairs.FastRopeEmbedding.Kernels
