import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.SourceSyntax

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Complete selected `_parallel_rebased_bwd_dq`, including the scaling
between its loops. The caller supplies the program coordinates. -/
def querySource
    (q k v do_ dz dq : RegionName)
    (i_bh i_c i_k i_v : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32, dz = tl.float32, dq = tl.float32
  p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")).to(b_q.dtype)
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_dq = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=(0, $(i_k) * $(BK)), block_shape=($(BTS), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_v_h),
    shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
    offsets=($(i_v) * $(BV), 0), block_shape=($(BV), $(BTS)), order=(0, 1))
  p_dz = dz + $(i_bh) * $(T) + $(i_c) * $(BTL) + tl.arange(0, $(BTL))
  b_dz = tl.load(p_dz, mask=(($(i_c) * $(BTL) + tl.arange(0, $(BTL))) < $(T)), other=0.0)
  for _i in range($(0), $(i_c) * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_ds = tl.dot(b_do, b_v, allow_tf32=false)
    if $(i_v) == $(0) {
      b_ds += b_dz[:, None]
    } else {
      b_ds = b_ds
    }
    b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=false)
    b_dq += tl.dot((2.0 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=false)
    p_k = tl.advance(p_k, [$(BTS), $(0)])
    p_v = tl.advance(p_v, [$(0), $(BTS)])
  }
  b_dq *= $((scale : ℝ))
  o_q = tl.arange(0, $(BTL))
  o_k = tl.arange(0, $(BTS))
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTS), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_v_h),
    shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
    offsets=($(i_v) * $(BV), $(i_c) * $(BTL)), block_shape=($(BV), $(BTS)), order=(0, 1))
  for _i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    m_s = o_q[:, None] >= o_k[None, :]
    b_ds = tl.dot(b_do, b_v, allow_tf32=false)
    if $(i_v) == $(0) {
      b_ds += b_dz[:, None]
    } else {
      b_ds = b_ds
    }
    b_ds = tl.where(m_s, b_ds, 0.0) * $((scale : ℝ))
    b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=false)
    b_s = tl.where(m_s, b_s, 0.0)
    b_dq += tl.dot((2.0 * b_ds * b_s).to(b_k.dtype), b_k, allow_tf32=false)
    p_k = tl.advance(p_k, [$(BTS), $(0)])
    p_v = tl.advance(p_v, [$(0), $(BTS)])
    o_k += $(BTS)
  }
  p_dq = tl.make_block_ptr(base=dq + $(((i_bh + B * H * i_v) * s_k_h : Nat)),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  tl.store(p_dq, (b_dq).to(p_dq.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
