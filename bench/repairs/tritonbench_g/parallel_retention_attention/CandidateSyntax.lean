import bench.repairs.tritonbench_g.parallel_retention_attention.QuerySyntax

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-! Fuse adjacent accumulator multiply and add assignments while retaining
loads, casts, explicit FP32 constants and the ordered primitive graph. -/
namespace FPStages
def fusedForwardSource
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
  b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_o = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_h[None, :]
    b_o = b_o * tl.math.exp2(b_b * tl.toReal($(BTS))) + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
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
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
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
end FPStages

namespace FPQuery
def fusedSource (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  let k := c.k
  let v := c.v
  let do_ := c.do_
  triton {
  tl.ports inputs=[do_, k, v], outputs=[dq]
  tl.region k = tl.float32, v = tl.float32, do_ = tl.float32, dq = tl.float32
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
}
end FPQuery

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
