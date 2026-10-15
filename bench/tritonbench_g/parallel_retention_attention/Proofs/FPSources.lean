import bench.tritonbench_g.parallel_retention_attention.ParallelRetentionAttention
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Complete original fp32 stages. Scalar full([],...) annotations materialize
Triton's default fp32 constants without adding a cast or arithmetic operation.
The original -5 constant and signed diagonal differences are retained; the
old mathematical transcription's 0-5 and clipped Nat differences are not used.
Natural address indices and widened signed index arithmetic require the
usual native index-range binding. Both loops, all dot options, all source
casts and unspecified block-pointer padding remain. No correctness claim
follows from these source definitions alone. -/
def forwardSource
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, o = tl.float32
  i_kv = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  NV = tl.cdiv($(V), $(BV))
  i_k = i_kv // NV
  i_v = i_kv % NV
  i_h = i_bh % $(H)
  b_b = tl.math.log2(tl.full([], 1.0, dtype=tl.float32) - tl.math.exp2(tl.full([], -5.0, dtype=tl.float32) - tl.toReal(i_h) * tl.full([], 1.0, dtype=tl.float32)))
  o_k = tl.arange(0, $(BTS))
  d_h = tl.math.exp2(tl.toReal($(BTS) - o_k) * b_b)
  p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=(i_c * $(BTL), i_k * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
    shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
    offsets=(i_k * $(BK), 0), block_shape=($(BK), $(BTS)), order=(0, 1))
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(0, i_v * $(BV)), block_shape=($(BTS), $(BV)), order=(1, 0))
  b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_o = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_h[None, :]
    b_o = b_o * tl.math.exp2(b_b * tl.toReal($(BTS)))
    b_o = b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTL))
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(BTL))) * b_b)
  b_o *= d_q[:, None]
  o_k = tl.arange(0, $(BTS))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
    shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
    offsets=(i_k * $(BK), i_c * $(BTL)), block_shape=($(BK), $(BTS)), order=(0, 1))
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTS), $(BV)), order=(1, 0))
  for _i in range(i_c * $(BTL), (i_c + $(1)) * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    m_s = o_q[:, None] >= o_k[None, :]
    d_s = tl.where(m_s, tl.math.exp2(tl.toReal((o_q[:, None]).to(tl.int64) - (o_k[None, :]).to(tl.int64)) * b_b), 0.0)
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_s
    b_o += tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
    o_k += $(BTS)
  }
  p_o = tl.make_block_ptr(base=o + (i_bh + $(B) * $(H) * i_k) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

def backwardSource
    (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32, dk = tl.float32, dv = tl.float32
  b_b = tl.math.log2(tl.full([], 1.0, dtype=tl.float32) - tl.math.exp2(tl.full([], -5.0, dtype=tl.float32) - tl.toReal($(i_h)) * tl.full([], 1.0, dtype=tl.float32)))
  d_b = tl.math.exp2(b_b * tl.toReal($(BTS)))
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_qk_h),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_dk = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  d_h = tl.math.exp2(tl.toReal($(BTL) - tl.arange(0, $(BTL))) * b_b)
  b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(BTS))) * b_b)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(T), $(BTS)) * $(BTS) - $(((i_c + 1) * BTL : Nat)), $(BTS)),
      $(1)) {
    i = tl.cdiv($(T), $(BTS)) * $(BTS) - $(BTS) - j * $(BTS)
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_vo_h),
      shape=($(V), $(T)), strides=($(s_vo_d), $(s_vo_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_do = (b_do * d_q[None, :]).to(b_do.dtype)
    b_dv *= d_b
    b_s = tl.dot((b_kd).to(b_q.dtype), b_q, allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk *= d_b
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  b_dk *= d_h[:, None] * $((scale : ℝ))
  b_dv *= $((scale : ℝ))
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTS))
  o_k = tl.arange(0, $(BTL))
  for i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_vo_h),
      shape=($(V), $(T)), strides=($(s_vo_d), $(s_vo_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    m_s = o_k[:, None] <= o_q[None, :]
    d_s = tl.where(m_s,
      tl.math.exp2(tl.toReal(-(o_k[:, None]).to(tl.int64) + (o_q[None, :]).to(tl.int64)) * (b_b).to(tl.float32)),
      0.0) * $((scale : ℝ))
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * d_s
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * d_s
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    o_q += $(BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((i_bh + B * H * i_v) * s_qk_h : Nat)),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((i_bh + B * H * i_k) * s_vo_h : Nat)),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

end
end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
