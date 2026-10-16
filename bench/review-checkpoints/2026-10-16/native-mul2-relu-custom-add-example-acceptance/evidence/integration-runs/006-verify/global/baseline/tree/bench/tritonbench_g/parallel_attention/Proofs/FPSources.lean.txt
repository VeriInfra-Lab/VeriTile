import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 5000000

/-- Original complete forward and backward source programs with fp32 storage
bound explicitly. Both loops, casts, TF32 options and unspecified block-pointer
tails are retained. These definitions alone assert no correctness theorem. -/
def forwardSource
    (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, o = tl.float32, z = tl.float32
  i_kv = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  NV = tl.cdiv($(V), $(BV))
  i_k = i_kv // NV
  i_v = i_kv % NV
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
  b_z = tl.zeros([$(BTL)], dtype=tl.float32)
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_s = b_s * b_s
    b_z += tl.sum(b_s, axis=1)
    b_o = b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTL))
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
    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_s = b_s * b_s
    b_s = tl.where(m_s, b_s, 0.0)
    b_z += tl.sum(b_s, axis=1)
    b_o += tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
    o_k += $(BTS)
  }
  p_o = tl.make_block_ptr(base=o + (i_bh + $(B) * $(H) * i_k) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  p_z = z + (i_bh + $(B) * $(H) * i_k) * $(T) + i_c * $(BTL) + tl.arange(0, $(BTL))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_z, (b_z).to(p_z.dtype.element_ty),
    mask=((i_c * $(BTL) + tl.arange(0, $(BTL))) < $(T)))
}

def backwardSource
    (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32, dz = tl.float32, dk = tl.float32, dv = tl.float32
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_dk = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(T), $(BTS)) * $(BTS) - $(((i_c + 1) * BTL : Nat)), $(BTS)),
      $(1)) {
    i = tl.cdiv($(T), $(BTS)) * $(BTS) - $(BTS) - j * $(BTS)
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
      shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
      shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    p_dz = dz + $(i_bh) * $(T) + i + tl.arange(0, $(BTS))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat))).to(b_q.dtype)
    b_dz = tl.load(p_dz, mask=((i + tl.arange(0, $(BTS))) < $(T)))
    b_s = tl.dot((b_k).to(b_q.dtype), b_q, allow_tf32=false) * $((scale : ℝ))
    b_s2 = b_s * b_s
    b_dv += tl.dot((b_s2).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * $((scale : ℝ))
    if $(i_v) == $(0) {
      b_ds += b_dz[None, :] * $((scale : ℝ))
    } else {
      b_ds = b_ds
    }
    b_dk += tl.dot((2.0 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTS))
  o_k = tl.arange(0, $(BTL))
  for i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
      shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
      shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    p_dz = dz + $(i_bh) * $(T) + i + tl.arange(0, $(BTS))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat))).to(b_q.dtype)
    b_dz = tl.load(p_dz, mask=((i + tl.arange(0, $(BTS))) < $(T)))
    m_s = o_k[:, None] <= o_q[None, :]
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * $((scale : ℝ))
    b_s2 = b_s * b_s
    b_s = tl.where(m_s, b_s, 0.0)
    b_s2 = tl.where(m_s, b_s2, 0.0)
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    if $(i_v) == $(0) {
      b_ds += b_dz[None, :]
    } else {
      b_ds = b_ds
    }
    b_ds = tl.where(m_s, b_ds, 0.0) * $((scale : ℝ))
    b_dv += tl.dot((b_s2).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk += tl.dot((2.0 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    o_q += $(BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((i_bh + B * H * i_v) * s_k_h : Nat)),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((i_bh + B * H * i_k) * s_v_h : Nat)),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
