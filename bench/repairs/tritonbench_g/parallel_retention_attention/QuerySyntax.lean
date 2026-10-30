/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.repairs.tritonbench_g.parallel_retention_attention.SourceSyntax
import bench.tritonbench_g.parallel_attention.Proofs.FPBackwardGraph

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- The original `_parallel_retention_bwd_dq` helper at fp32 storage.
The shared configuration's q/o/z/dk/dv/dz fields are unused. Keep the two
different scale sites, both loops, eager signed mask arithmetic, and explicit zero
padding. Nat addresses and widened signed differences need a separate native
integer binding; this is not an assertion about arbitrary hardware launches. -/
def source (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
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
}

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
