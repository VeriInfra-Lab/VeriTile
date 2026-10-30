import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Historical literal-specialized proof reference. These complete ordinary
quotations preserve the prior AST, including its scalar-model scale marker.
Actual kernel entry points use Runtime.forward/backward in Kernels.lean, with
an explicit tl.param and Invocation.arguments. RuntimeBridge proves their link.
This file is proof material, not an alternate authoritative runtime source. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Source
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000

def forward
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) :
    ComputeKernel := if IS_HEADWISE_BETA then triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_beta += $(V)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
} else triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(T)
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    b_beta = tl.load(p_beta).to(tl.float32)
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_beta += $(1)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}

def backward
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) :
    ComputeKernel := if IS_HEADWISE_BETA then triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, dht = tl.float32, dh0 = tl.float32,
    do_ = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32, dbeta = tl.float32, h0 = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(K)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(V)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_dbeta = dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_DHT {
    p_ht = dht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
  }
  for _i in range($(T)) {
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    d_h += b_q[:, None] * b_do[None, :]
    d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
    d_v = tl.sum(d_h * b_k[:, None], axis=0)
    d_beta = d_v * b_v
    d_v = d_v * b_beta
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
    tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty), mask=mask_bv)
    d_h -= b_k[:, None] * d_v[None, :]
    p_do -= $(V)
    p_q -= $(K)
    p_k -= $(K)
    p_v -= $(V)
    p_dk -= $(K)
    p_dv -= $(V)
    p_dbeta -= $(V)
    p_beta -= $(V)
  }
  if USE_DH0 {
    p_dh0 = dh0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    tl.store(p_dh0, (d_h).to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])
  }
  tl.debug_barrier()
  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for i in range($(0), $(T), $(1)) {
    d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
    d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
    d_k -= tl.sum(d_v[None, :] * h, axis=1)
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    b_v *= b_beta
    h += b_k[:, None] * b_v[None, :]
    _d_q = h * b_do[None, :]
    d_q = tl.sum(_d_q, axis=1) * $(scale)
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
    p_k += $(K)
    p_do += $(V)
    p_v += $(V)
    p_dk += $(K)
    p_dv += $(V)
    p_dq += $(K)
    p_beta += $(V)
  }
} else triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, dht = tl.float32, dh0 = tl.float32,
    do_ = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32, dbeta = tl.float32, h0 = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(K)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(V)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_beta = beta + i_bh * $(T) + ($(T) - $(1))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_dbeta = dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + ($(T) - $(1))
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_DHT {
    p_ht = dht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
  }
  for _i in range($(T)) {
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta).to(tl.float32)
    d_h += b_q[:, None] * b_do[None, :]
    d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
    d_v = tl.sum(d_h * b_k[:, None], axis=0)
    d_beta = tl.sum(d_v * b_v)
    d_v = d_v * b_beta
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
    tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty))
    d_h -= b_k[:, None] * d_v[None, :]
    p_do -= $(V)
    p_q -= $(K)
    p_k -= $(K)
    p_v -= $(V)
    p_dk -= $(K)
    p_dv -= $(V)
    p_dbeta -= $(1)
    p_beta -= $(1)
  }
  if USE_DH0 {
    p_dh0 = dh0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    tl.store(p_dh0, (d_h).to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])
  }
  tl.debug_barrier()
  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(T)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for i in range($(0), $(T), $(1)) {
    d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
    d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
    d_k -= tl.sum(d_v[None, :] * h, axis=1)
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta).to(tl.float32)
    b_v *= b_beta
    h += b_k[:, None] * b_v[None, :]
    _d_q = h * b_do[None, :]
    d_q = tl.sum(_d_q, axis=1) * $(scale)
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
    p_k += $(K)
    p_do += $(V)
    p_v += $(V)
    p_dk += $(K)
    p_dv += $(V)
    p_dq += $(K)
    p_beta += $(1)
  }
}

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Source
