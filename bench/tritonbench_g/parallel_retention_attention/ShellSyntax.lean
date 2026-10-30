import bench.tritonbench_g.parallel_retention_attention.QuerySyntax
import bench.tritonbench_g.parallel_retention_attention.CandidateSyntax

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-! Complete readable backward DQ/barrier/DKV shells. Caller-decoded program
coordinates and the two helper stores remain explicit. -/
def shellSource (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  let q := c.q
  let k := c.k
  let v := c.v
  let do_ := c.do_
  let dk := c.dk
  let dv := c.dv
  triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32
  p_do = tl.make_block_ptr(base=do_ + $(c.i_bh * c.s_vo_h),
    shape=($(c.T), $(c.V)), strides=($(c.s_vo_t), $(c.s_vo_d)),
    offsets=($(c.i_c * c.BTL), $(c.i_v * c.BV)), block_shape=($(c.BTL), $(c.BV)), order=(1, 0))
  b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_dq = tl.zeros([$(c.BTL), $(c.BK)], dtype=tl.float32)
  p_k = tl.make_block_ptr(base=k + $(c.i_bh * c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(0), $(c.i_k * c.BK)), block_shape=($(c.BTS), $(c.BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(c.i_bh * c.s_vo_h),
    shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
    offsets=($(c.i_v * c.BV), $(0)), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
  b_b = tl.math.log2(tl.full([], 1.0, dtype=tl.float32) - tl.math.exp2(tl.full([], -5.0, dtype=tl.float32) - tl.toReal($(c.i_h)) * tl.full([], 1.0, dtype=tl.float32)))
  d_b = tl.math.exp2(b_b * tl.toReal($(c.BTS)))
  d_h = tl.math.exp2(tl.toReal($(c.BTS) - tl.arange(0, $(c.BTS))) * b_b)
  for _i in range($(0), $(c.i_c * c.BTL), $(c.BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_ds = tl.dot(b_do, b_v, allow_tf32=false) * d_h[None, :]
    b_dq *= d_b
    b_dq += tl.dot((b_ds).to(b_v.dtype), b_k, allow_tf32=false)
    p_k = tl.advance(p_k, [$(c.BTS), $(0)])
    p_v = tl.advance(p_v, [$(0), $(c.BTS)])
  }
  b_dq *= tl.math.exp2(tl.toReal(tl.arange(0, $(c.BTL))) * b_b)[:, None] * $((c.scale : ℝ))
  o_q = tl.arange(0, $(c.BTL))
  o_k = tl.arange(0, $(c.BTS))
  p_k = tl.make_block_ptr(base=k + $(c.i_bh * c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c * c.BTL), $(c.i_k * c.BK)), block_shape=($(c.BTS), $(c.BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(c.i_bh * c.s_vo_h),
    shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
    offsets=($(c.i_v * c.BV), $(c.i_c * c.BTL)), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
  for _i in range($(c.i_c * c.BTL), $((c.i_c + 1) * c.BTL), $(c.BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    m_s = o_q[:, None] >= o_k[None, :]
    d_s = tl.where(m_s, tl.math.exp2(tl.toReal((o_q[:, None]).to(tl.int64) - (o_k[None, :]).to(tl.int64)) * b_b), 0.0)
    b_ds = tl.dot(b_do, b_v, allow_tf32=false) * d_s * $((c.scale : ℝ))
    b_dq += tl.dot((b_ds).to(b_k.dtype), b_k, allow_tf32=false)
    p_k = tl.advance(p_k, [$(c.BTS), $(0)])
    p_v = tl.advance(p_v, [$(0), $(c.BTS)])
    o_k += $(c.BTS)
  }
  p_dq = tl.make_block_ptr(base=dq + $((c.i_bh + c.B * c.H * c.i_v) * c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c * c.BTL), $(c.i_k * c.BK)), block_shape=($(c.BTL), $(c.BK)), order=(1, 0))
  tl.store(p_dq, (b_dq).to(p_dq.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.debug_barrier()
  b_b = tl.math.log2(tl.full([], 1.0, dtype=tl.float32) - tl.math.exp2(tl.full([], -5.0, dtype=tl.float32) - tl.toReal($(c.i_h)) * tl.full([], 1.0, dtype=tl.float32)))
  d_b = tl.math.exp2(b_b * tl.toReal($(c.BTS)))
  p_k = tl.make_block_ptr(base=k + $(c.i_bh) * $(c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_k) * $(c.BK)), block_shape=($(c.BTL), $(c.BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(c.i_bh) * $(c.s_vo_h),
    shape=($(c.T), $(c.V)), strides=($(c.s_vo_t), $(c.s_vo_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_v) * $(c.BV)), block_shape=($(c.BTL), $(c.BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_dk = tl.zeros([$(c.BTL), $(c.BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(c.BTL), $(c.BV)], dtype=tl.float32)
  d_h = tl.math.exp2(tl.toReal($(c.BTL) - tl.arange(0, $(c.BTL))) * b_b)
  b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(c.BTS))) * b_b)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(c.T), $(c.BTS)) * $(c.BTS) - $(((c.i_c + 1) * c.BTL : Nat)), $(c.BTS)),
      $(1)) {
    i = tl.cdiv($(c.T), $(c.BTS)) * $(c.BTS) - $(c.BTS) - j * $(c.BTS)
    p_q = tl.make_block_ptr(base=q + $(c.i_bh) * $(c.s_qk_h),
      shape=($(c.K), $(c.T)), strides=($(c.s_qk_d), $(c.s_qk_t)),
      offsets=($(c.i_k) * $(c.BK), i), block_shape=($(c.BK), $(c.BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(c.i_bh) * $(c.s_vo_h),
      shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
      offsets=($(c.i_v) * $(c.BV), i), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_do = (b_do * d_q[None, :]).to(b_do.dtype)
    b_dv *= d_b
    b_s = tl.dot((b_kd).to(b_q.dtype), b_q, allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk *= d_b
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  b_dk *= d_h[:, None] * $((c.scale : ℝ))
  b_dv *= $((c.scale : ℝ))
  tl.debug_barrier()
  o_q = tl.arange(0, $(c.BTS))
  o_k = tl.arange(0, $(c.BTL))
  for i in range($(c.i_c) * $(c.BTL), $(((c.i_c + 1) * c.BTL : Nat)), $(c.BTS)) {
    p_q = tl.make_block_ptr(base=q + $(c.i_bh) * $(c.s_qk_h),
      shape=($(c.K), $(c.T)), strides=($(c.s_qk_d), $(c.s_qk_t)),
      offsets=($(c.i_k) * $(c.BK), i), block_shape=($(c.BK), $(c.BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(c.i_bh) * $(c.s_vo_h),
      shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
      offsets=($(c.i_v) * $(c.BV), i), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
    m_s = o_k[:, None] <= o_q[None, :]
    d_s = tl.where(m_s,
      tl.math.exp2(tl.toReal(-(o_k[:, None]).to(tl.int64) + (o_q[None, :]).to(tl.int64)) * (b_b).to(tl.float32)),
      0.0) * $((c.scale : ℝ))
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * d_s
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * d_s
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    o_q += $(c.BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((c.i_bh + c.B * c.H * c.i_v) * c.s_qk_h : Nat)),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_k) * $(c.BK)), block_shape=($(c.BTL), $(c.BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((c.i_bh + c.B * c.H * c.i_k) * c.s_vo_h : Nat)),
    shape=($(c.T), $(c.V)), strides=($(c.s_vo_t), $(c.s_vo_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_v) * $(c.BV)), block_shape=($(c.BTL), $(c.BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

def fusedShellSource (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  let q := c.q
  let k := c.k
  let v := c.v
  let do_ := c.do_
  let dk := c.dk
  let dv := c.dv
  triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32
  p_do = tl.make_block_ptr(base=do_ + $(c.i_bh * c.s_vo_h),
    shape=($(c.T), $(c.V)), strides=($(c.s_vo_t), $(c.s_vo_d)),
    offsets=($(c.i_c * c.BTL), $(c.i_v * c.BV)), block_shape=($(c.BTL), $(c.BV)), order=(1, 0))
  b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_dq = tl.zeros([$(c.BTL), $(c.BK)], dtype=tl.float32)
  p_k = tl.make_block_ptr(base=k + $(c.i_bh * c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(0), $(c.i_k * c.BK)), block_shape=($(c.BTS), $(c.BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(c.i_bh * c.s_vo_h),
    shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
    offsets=($(c.i_v * c.BV), $(0)), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
  b_b = tl.math.log2(tl.full([], 1.0, dtype=tl.float32) - tl.math.exp2(tl.full([], -5.0, dtype=tl.float32) - tl.toReal($(c.i_h)) * tl.full([], 1.0, dtype=tl.float32)))
  d_b = tl.math.exp2(b_b * tl.toReal($(c.BTS)))
  d_h = tl.math.exp2(tl.toReal($(c.BTS) - tl.arange(0, $(c.BTS))) * b_b)
  for _i in range($(0), $(c.i_c * c.BTL), $(c.BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_ds = tl.dot(b_do, b_v, allow_tf32=false) * d_h[None, :]
    b_dq = b_dq * d_b + tl.dot((b_ds).to(b_v.dtype), b_k, allow_tf32=false)
    p_k = tl.advance(p_k, [$(c.BTS), $(0)])
    p_v = tl.advance(p_v, [$(0), $(c.BTS)])
  }
  b_dq *= tl.math.exp2(tl.toReal(tl.arange(0, $(c.BTL))) * b_b)[:, None] * $((c.scale : ℝ))
  o_q = tl.arange(0, $(c.BTL))
  o_k = tl.arange(0, $(c.BTS))
  p_k = tl.make_block_ptr(base=k + $(c.i_bh * c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c * c.BTL), $(c.i_k * c.BK)), block_shape=($(c.BTS), $(c.BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(c.i_bh * c.s_vo_h),
    shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
    offsets=($(c.i_v * c.BV), $(c.i_c * c.BTL)), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
  for _i in range($(c.i_c * c.BTL), $((c.i_c + 1) * c.BTL), $(c.BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    m_s = o_q[:, None] >= o_k[None, :]
    d_s = tl.where(m_s, tl.math.exp2(tl.toReal((o_q[:, None]).to(tl.int64) - (o_k[None, :]).to(tl.int64)) * b_b), 0.0)
    b_ds = tl.dot(b_do, b_v, allow_tf32=false) * d_s * $((c.scale : ℝ))
    b_dq += tl.dot((b_ds).to(b_k.dtype), b_k, allow_tf32=false)
    p_k = tl.advance(p_k, [$(c.BTS), $(0)])
    p_v = tl.advance(p_v, [$(0), $(c.BTS)])
    o_k += $(c.BTS)
  }
  p_dq = tl.make_block_ptr(base=dq + $((c.i_bh + c.B * c.H * c.i_v) * c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c * c.BTL), $(c.i_k * c.BK)), block_shape=($(c.BTL), $(c.BK)), order=(1, 0))
  tl.store(p_dq, (b_dq).to(p_dq.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.debug_barrier()
  b_b = tl.math.log2(tl.full([], 1.0, dtype=tl.float32) - tl.math.exp2(tl.full([], -5.0, dtype=tl.float32) - tl.toReal($(c.i_h)) * tl.full([], 1.0, dtype=tl.float32)))
  d_b = tl.math.exp2(b_b * tl.toReal($(c.BTS)))
  p_k = tl.make_block_ptr(base=k + $(c.i_bh) * $(c.s_qk_h),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_k) * $(c.BK)), block_shape=($(c.BTL), $(c.BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(c.i_bh) * $(c.s_vo_h),
    shape=($(c.T), $(c.V)), strides=($(c.s_vo_t), $(c.s_vo_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_v) * $(c.BV)), block_shape=($(c.BTL), $(c.BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_dk = tl.zeros([$(c.BTL), $(c.BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(c.BTL), $(c.BV)], dtype=tl.float32)
  d_h = tl.math.exp2(tl.toReal($(c.BTL) - tl.arange(0, $(c.BTL))) * b_b)
  b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(c.BTS))) * b_b)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(c.T), $(c.BTS)) * $(c.BTS) - $(((c.i_c + 1) * c.BTL : Nat)), $(c.BTS)),
      $(1)) {
    i = tl.cdiv($(c.T), $(c.BTS)) * $(c.BTS) - $(c.BTS) - j * $(c.BTS)
    p_q = tl.make_block_ptr(base=q + $(c.i_bh) * $(c.s_qk_h),
      shape=($(c.K), $(c.T)), strides=($(c.s_qk_d), $(c.s_qk_t)),
      offsets=($(c.i_k) * $(c.BK), i), block_shape=($(c.BK), $(c.BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(c.i_bh) * $(c.s_vo_h),
      shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
      offsets=($(c.i_v) * $(c.BV), i), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_do = (b_do * d_q[None, :]).to(b_do.dtype)
    b_dv *= d_b
    b_s = tl.dot((b_kd).to(b_q.dtype), b_q, allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk *= d_b
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  b_dk *= d_h[:, None] * $((c.scale : ℝ))
  b_dv *= $((c.scale : ℝ))
  tl.debug_barrier()
  o_q = tl.arange(0, $(c.BTS))
  o_k = tl.arange(0, $(c.BTL))
  for i in range($(c.i_c) * $(c.BTL), $(((c.i_c + 1) * c.BTL : Nat)), $(c.BTS)) {
    p_q = tl.make_block_ptr(base=q + $(c.i_bh) * $(c.s_qk_h),
      shape=($(c.K), $(c.T)), strides=($(c.s_qk_d), $(c.s_qk_t)),
      offsets=($(c.i_k) * $(c.BK), i), block_shape=($(c.BK), $(c.BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(c.i_bh) * $(c.s_vo_h),
      shape=($(c.V), $(c.T)), strides=($(c.s_vo_d), $(c.s_vo_t)),
      offsets=($(c.i_v) * $(c.BV), i), block_shape=($(c.BV), $(c.BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
    m_s = o_k[:, None] <= o_q[None, :]
    d_s = tl.where(m_s,
      tl.math.exp2(tl.toReal(-(o_k[:, None]).to(tl.int64) + (o_q[None, :]).to(tl.int64)) * (b_b).to(tl.float32)),
      0.0) * $((c.scale : ℝ))
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * d_s
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * d_s
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    o_q += $(c.BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((c.i_bh + c.B * c.H * c.i_v) * c.s_qk_h : Nat)),
    shape=($(c.T), $(c.K)), strides=($(c.s_qk_t), $(c.s_qk_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_k) * $(c.BK)), block_shape=($(c.BTL), $(c.BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((c.i_bh + c.B * c.H * c.i_k) * c.s_vo_h : Nat)),
    shape=($(c.T), $(c.V)), strides=($(c.s_vo_t), $(c.s_vo_d)),
    offsets=($(c.i_c) * $(c.BTL), $(c.i_v) * $(c.BV)), block_shape=($(c.BTL), $(c.BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}


end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward
