import bench.tritonbench_g.chunk_gla_fwd.SourceCompatibility

/-! Complete typed DSL bodies used by the repaired five-kernel task.
The selected FP32 originals and existing commuting candidates have complete
ordinary quotations. Historical mixed-storage branches retain compatibility
macros in SourceCompatibility.lean; their public source interfaces are unchanged.
All 14 block loads have explicit zero padding. Five value-identity compiler
hint expressions remain in this DSL representation; the repaired Python
removes those invalid alignment hints. checks/source_binding.py compares
all five complete selected bodies after erasing only those hint wrappers,
static FP32 dtype spellings, and syntax-only spelling differences.
The corrected finite scratch shape, allocation order, grids and both public
entry points are modeled separately in Proofs/HostMetadata and Proofs/FPHost.
-/

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_A_kernel_intra_sub_inter.
Complete inter-subblock source. Query, key and
 gate storage independently select fp32/fp16/bf16. Both source early returns
 are transcribed directly; the candidate commutes only the fp32
 carry/dot addition in the complete K loop. Intrinsic exp, gate subtraction
 precision, default dot options, explicit zero block padding and output casting
 are retained. The flat gate reference uses rowStart*K; the other gate loads
 use s_k_t, exactly as in the source. No equal-stride assumption is inserted. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter
open Lean VeriTile.Triton
abbrev Storage := TypedOutput.Storage
/-- Selected FP32 original, with its complete ordinary source body. -/
def fp32Original (out_dtype : FloatDType) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,g], outputs=[A]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, A = OUT_DTYPE
    i_t = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_i = i_c // $(NC)
    i_j = i_c % $(NC)
    if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
    if i_i <= i_j { return }
    b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
    for i_k in range(tl.cdiv($(K), $(BK))) {
      o_k = i_k * $(BK) + tl.arange(0, $(BK))
      m_k = o_k < $(K)
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(K), $(T)), strides=($(1), $(s_k_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_j * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_gk = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
        shape=($(K), $(T)), strides=($(1), $(s_k_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_j * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) +
        (i_t * $(BT) + i_i * $(BC)) * $(K) + o_k, $(BK)), $(BK))
      b_gn = tl.load(p_gn, mask=m_k, other=0)
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * $((scale : ℝ))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_gk = tl.load(p_gk, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
      b_A += tl.dot(b_qg, b_kg)
    }
    p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
      shape=($(T), $(BT)), strides=($(BT), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), i_j * $(BC)),
      block_shape=($(BC), $(BC)), order=(1, 0))
    tl.store(p_A, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
  }

/-- Retained candidate; only its existing local operand swap differs. -/
def fp32DotFirst (out_dtype : FloatDType) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,g], outputs=[A]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, A = OUT_DTYPE
    i_t = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_i = i_c // $(NC)
    i_j = i_c % $(NC)
    if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
    if i_i <= i_j { return }
    b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
    for i_k in range(tl.cdiv($(K), $(BK))) {
      o_k = i_k * $(BK) + tl.arange(0, $(BK))
      m_k = o_k < $(K)
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(K), $(T)), strides=($(1), $(s_k_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_j * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_gk = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
        shape=($(K), $(T)), strides=($(1), $(s_k_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_j * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) +
        (i_t * $(BT) + i_i * $(BC)) * $(K) + o_k, $(BK)), $(BK))
      b_gn = tl.load(p_gn, mask=m_k, other=0)
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * $((scale : ℝ))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_gk = tl.load(p_gk, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
      b_A = tl.dot(b_qg, b_kg) + b_A
    }
    p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
      shape=($(T), $(BT)), strides=($(BT), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), i_j * $(BC)),
      block_shape=($(BC), $(BC)), order=(1, 0))
    tl.store(p_A, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
  }

set_option maxHeartbeats 5000000 in
def source (qStorage kStorage gStorage : Storage) (swapped : Bool) :=
  match qStorage,kStorage,gStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then fp32DotFirst
      else fp32Original
  | .fp32, .fp32, .fp16 =>
      if swapped then glaInterSource%[tl.float32, tl.float32, tl.float16, dotFirst]
      else glaInterSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaInterSource%[tl.float32, tl.float32, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaInterSource%[tl.float32, tl.float16, tl.float32, dotFirst]
      else glaInterSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaInterSource%[tl.float32, tl.float16, tl.float16, dotFirst]
      else glaInterSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaInterSource%[tl.float32, tl.float16, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaInterSource%[tl.float32, tl.bfloat16, tl.float32, dotFirst]
      else glaInterSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaInterSource%[tl.float32, tl.bfloat16, tl.float16, dotFirst]
      else glaInterSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaInterSource%[tl.float32, tl.bfloat16, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaInterSource%[tl.float16, tl.float32, tl.float32, dotFirst]
      else glaInterSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaInterSource%[tl.float16, tl.float32, tl.float16, dotFirst]
      else glaInterSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaInterSource%[tl.float16, tl.float32, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaInterSource%[tl.float16, tl.float16, tl.float32, dotFirst]
      else glaInterSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaInterSource%[tl.float16, tl.float16, tl.float16, dotFirst]
      else glaInterSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaInterSource%[tl.float16, tl.float16, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaInterSource%[tl.float16, tl.bfloat16, tl.float32, dotFirst]
      else glaInterSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaInterSource%[tl.float16, tl.bfloat16, tl.float16, dotFirst]
      else glaInterSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaInterSource%[tl.float16, tl.bfloat16, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.float32, tl.float32, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.float32, tl.float16, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.float32, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.float16, tl.float32, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.float16, tl.float16, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.float16, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.bfloat16, tl.float32, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.bfloat16, tl.float16, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaInterSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, dotFirst]
      else glaInterSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
def original (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.false
def dotFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra. The candidate
commutes only scaling inside the causal where. Query/gate block loads keep
unspecified padding; flat key/gate loads keep other=0 and the explicit fp32
casts. Their K stride is distinct from the query/gate matrix time stride.
The output A is the fp32 intermediate allocated by the original host. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra
open Lean VeriTile.Triton
abbrev Storage := TypedOutput.Storage
/-- Selected FP32 original, with its complete ordinary source body. -/
def fp32Original (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,g], outputs=[A]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, A = tl.float32
    i_t = tl.program_id(0)
    i_i = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_j = i_i
    if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
    o_i = tl.arange(0, $(BC))
    o_k = tl.arange(0, $(BK))
    o_A = i_bh * $(T) * $(BT) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BT) + i_j * $(BC)
    m_k = o_k < $(K)
    m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), $(0)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), $(0)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
    for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
      b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
      b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
      b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
      b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0)
      tl.store(A + o_A + j, b_A, mask=m_A)
      p_k += $(K)
      p_gk += $(K)
    }
  }

/-- Retained candidate; only its existing local operand swap differs. -/
def fp32ScaleFirst (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,g], outputs=[A]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, A = tl.float32
    i_t = tl.program_id(0)
    i_i = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_j = i_i
    if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
    o_i = tl.arange(0, $(BC))
    o_k = tl.arange(0, $(BK))
    o_A = i_bh * $(T) * $(BT) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BT) + i_j * $(BC)
    m_k = o_k < $(K)
    m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), $(0)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), $(0)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
    for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
      b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
      b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
      b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
      b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0)
      tl.store(A + o_A + j, b_A, mask=m_A)
      p_k += $(K)
      p_gk += $(K)
    }
  }

set_option maxHeartbeats 5000000 in
def source (qStorage kStorage gStorage : Storage) (swapped : Bool) :=
  match qStorage,kStorage,gStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then fp32ScaleFirst
      else fp32Original
  | .fp32, .fp32, .fp16 =>
      if swapped then glaIntraSource%[tl.float32, tl.float32, tl.float16, scaleFirst]
      else glaIntraSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaIntraSource%[tl.float32, tl.float32, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaIntraSource%[tl.float32, tl.float16, tl.float32, scaleFirst]
      else glaIntraSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaIntraSource%[tl.float32, tl.float16, tl.float16, scaleFirst]
      else glaIntraSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaIntraSource%[tl.float32, tl.float16, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaIntraSource%[tl.float32, tl.bfloat16, tl.float32, scaleFirst]
      else glaIntraSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaIntraSource%[tl.float32, tl.bfloat16, tl.float16, scaleFirst]
      else glaIntraSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaIntraSource%[tl.float32, tl.bfloat16, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaIntraSource%[tl.float16, tl.float32, tl.float32, scaleFirst]
      else glaIntraSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaIntraSource%[tl.float16, tl.float32, tl.float16, scaleFirst]
      else glaIntraSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaIntraSource%[tl.float16, tl.float32, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaIntraSource%[tl.float16, tl.float16, tl.float32, scaleFirst]
      else glaIntraSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaIntraSource%[tl.float16, tl.float16, tl.float16, scaleFirst]
      else glaIntraSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaIntraSource%[tl.float16, tl.float16, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaIntraSource%[tl.float16, tl.bfloat16, tl.float32, scaleFirst]
      else glaIntraSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaIntraSource%[tl.float16, tl.bfloat16, tl.float16, scaleFirst]
      else glaIntraSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaIntraSource%[tl.float16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.float32, tl.float32, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.float32, tl.float16, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.float32, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.float16, tl.float32, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.float16, tl.float16, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.float16, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.bfloat16, tl.float32, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.bfloat16, tl.float16, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaIntraSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else glaIntraSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
def original (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.false
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra_split. The candidate
commutes only scaling inside the causal where. Query/gate block loads keep
unspecified padding; flat key/gate loads keep other=0 and the explicit fp32
casts. Their K stride is distinct from the query/gate matrix time stride.
The output A is the fp32 split intermediate. Its source slice stride T*BC
is retained; the host allocation must separately cover those addresses. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit
open Lean VeriTile.Triton
abbrev Storage := TypedOutput.Storage
/-- Selected FP32 original, with its complete ordinary source body. -/
def fp32Original (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,g], outputs=[A]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, A = tl.float32
    i_k = tl.program_id(0)
    i_tc = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_t = i_tc // $(NC)
    i_i = i_tc % $(NC)
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
    o_i = tl.arange(0, $(BC))
    o_k = i_k * $(BK) + tl.arange(0, $(BK))
    o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
    m_k = o_k < $(K)
    m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
    for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
      b_A = tl.zeros([$(BC)], dtype=tl.float32)
      b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
      b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
      b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
      b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0)
      tl.store(A + o_A + j, b_A, mask=m_A)
      p_k += $(K)
      p_gk += $(K)
    }
  }

/-- Retained candidate; only its existing local operand swap differs. -/
def fp32ScaleFirst (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,k,g], outputs=[A]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, A = tl.float32
    i_k = tl.program_id(0)
    i_tc = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_t = i_tc // $(NC)
    i_i = i_tc % $(NC)
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
    o_i = tl.arange(0, $(BC))
    o_k = i_k * $(BK) + tl.arange(0, $(BK))
    o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
    m_k = o_k < $(K)
    m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
    for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
      b_A = tl.zeros([$(BC)], dtype=tl.float32)
      b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
      b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
      b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
      b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0)
      tl.store(A + o_A + j, b_A, mask=m_A)
      p_k += $(K)
      p_gk += $(K)
    }
  }

set_option maxHeartbeats 5000000 in
def source (qStorage kStorage gStorage : Storage) (swapped : Bool) :=
  match qStorage,kStorage,gStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then fp32ScaleFirst
      else fp32Original
  | .fp32, .fp32, .fp16 =>
      if swapped then glaSplitSource%[tl.float32, tl.float32, tl.float16, scaleFirst]
      else glaSplitSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaSplitSource%[tl.float32, tl.float32, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaSplitSource%[tl.float32, tl.float16, tl.float32, scaleFirst]
      else glaSplitSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaSplitSource%[tl.float32, tl.float16, tl.float16, scaleFirst]
      else glaSplitSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaSplitSource%[tl.float32, tl.float16, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaSplitSource%[tl.float32, tl.bfloat16, tl.float32, scaleFirst]
      else glaSplitSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaSplitSource%[tl.float32, tl.bfloat16, tl.float16, scaleFirst]
      else glaSplitSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaSplitSource%[tl.float32, tl.bfloat16, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaSplitSource%[tl.float16, tl.float32, tl.float32, scaleFirst]
      else glaSplitSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaSplitSource%[tl.float16, tl.float32, tl.float16, scaleFirst]
      else glaSplitSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaSplitSource%[tl.float16, tl.float32, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaSplitSource%[tl.float16, tl.float16, tl.float32, scaleFirst]
      else glaSplitSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaSplitSource%[tl.float16, tl.float16, tl.float16, scaleFirst]
      else glaSplitSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaSplitSource%[tl.float16, tl.float16, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaSplitSource%[tl.float16, tl.bfloat16, tl.float32, scaleFirst]
      else glaSplitSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaSplitSource%[tl.float16, tl.bfloat16, tl.float16, scaleFirst]
      else glaSplitSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaSplitSource%[tl.float16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.float32, tl.float32, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.float32, tl.float16, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.float32, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.float16, tl.float32, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.float16, tl.float16, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.float16, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.bfloat16, tl.float32, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.bfloat16, tl.float16, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaSplitSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else glaSplitSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
def original (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.false
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra_merge. The host
allocates its temporary input in fp32; output storage stays explicit. The source
early return is transcribed directly. Runtime head count,
all NK contributions, omitted padding and output conversion are retained. The
candidate commutes only the fp32 carry/load addition inside the loop. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge
open Lean VeriTile.Triton
/-- Selected FP32 original, with its complete ordinary source body. -/
def original (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[A], outputs=[A2]
    tl.region A = tl.float32, A2 = OUT_DTYPE
    i_t = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
    for i_k in range($(0), $(NK)) {
      p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
        shape=($(T), $(BC)), strides=($(BC), $(1)),
        offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
        block_shape=($(BC), $(BC)), order=(1, 0))
      b_A += tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero")
    }
    p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
      shape=($(T), $(BT)), strides=($(BT), $(1)),
      offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
      block_shape=($(BC), $(BC)), order=(1, 0))
    tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
  }

/-- Retained candidate; only its existing local operand swap differs. -/
def loadFirst (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[A], outputs=[A2]
    tl.region A = tl.float32, A2 = OUT_DTYPE
    i_t = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
    for i_k in range($(0), $(NK)) {
      p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
        shape=($(T), $(BC)), strides=($(BC), $(1)),
        offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
        block_shape=($(BC), $(BC)), order=(1, 0))
      b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero") + b_A
    }
    p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
      shape=($(T), $(BT)), strides=($(BT), $(1)),
      offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
      block_shape=($(BC), $(BC)), order=(1, 0))
    tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
  }

end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_kernel_o.
Complete typed source of the output consumer.
Q, V and gate storage independently select fp32/fp16/bf16. History and output
storage remain explicit dtype parameters; A is fp32, as allocated by the host.
Query scale and exp-gate results are cast back to query storage. H is cast to
that same storage before the default-option inter-chunk dot; A is cast to V
storage before the TF32-disabled final dot. All block loads use explicit zero padding.
The candidate commutes only the final fp32 carry/intra-chunk dot addition.
All dimensions, strides and program IDs are symbolic. This source covers the
output consumer; the four A producers have separate proof obligations. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton

/-- Selected FP32 original, with its complete ordinary source body. -/
def fp32Original (dtype out_dtype : FloatDType) (q v g h o A : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,v,g,h,A], outputs=[o]
    tl.region q = tl.float32, v = tl.float32, g = tl.float32, h = DTYPE,
      o = OUT_DTYPE, A = tl.float32
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
    b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
    for i_k in range(tl.cdiv($(K), $(BK))) {
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
        shape=($(K), $(V)), strides=($(s_h_t), $(1)),
        offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = (b_q * $((scale : ℝ))).to(tl.float32)
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_qg = (b_q * tl.exp(b_g)).to(tl.float32)
      b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
      if i_k >= $(0) {
        b_o += tl.dot(b_qg, (b_h).to(tl.float32))
      }
    }
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
      shape=($(T), $(V)), strides=($(s_v_t), $(1)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
      shape=($(T), $(V)), strides=($(s_v_t), $(1)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
      shape=($(T), $(BT)), strides=($(BT), $(1)),
      offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_A = (tl.where(m_s, b_A, 0.0)).to(tl.float32)
    b_o += tl.dot(b_A, b_v, allow_tf32=false)
    tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
  }

/-- Retained candidate; only its existing local operand swap differs. -/
def fp32SumLeft (dtype out_dtype : FloatDType) (q v g h o A : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[q,v,g,h,A], outputs=[o]
    tl.region q = tl.float32, v = tl.float32, g = tl.float32, h = DTYPE,
      o = OUT_DTYPE, A = tl.float32
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
    b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
    for i_k in range(tl.cdiv($(K), $(BK))) {
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
        shape=($(K), $(V)), strides=($(s_h_t), $(1)),
        offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = (b_q * $((scale : ℝ))).to(tl.float32)
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_qg = (b_q * tl.exp(b_g)).to(tl.float32)
      b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
      if i_k >= $(0) {
        b_o += tl.dot(b_qg, (b_h).to(tl.float32))
      }
    }
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
      shape=($(T), $(V)), strides=($(s_v_t), $(1)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
      shape=($(T), $(V)), strides=($(s_v_t), $(1)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
      shape=($(T), $(BT)), strides=($(BT), $(1)),
      offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_A = (tl.where(m_s, b_A, 0.0)).to(tl.float32)
    b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o
    tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
  }

set_option maxHeartbeats 4000000 in
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match queryStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then fp32SumLeft
      else fp32Original
  | .fp32, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]

def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false

def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true

end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput
