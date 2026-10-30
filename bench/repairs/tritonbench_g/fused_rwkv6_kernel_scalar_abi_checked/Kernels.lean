import VeriTile.Triton
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Kernels

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
open VeriTile.Triton

/-- Complete selected register-loop source: repaired scratch pitch, hoisted U,
both traversal directions, optional seed and optional final store. The natural
pointer channel clips unobserved negative offsets at T=0/final backedges;
the active-address and dead-register proofs justify that projection separately. -/
def kernel (c : FusedRwkv6Kernel.Shape) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  triton {
    tl.param scale : tl.float32
    tl.region q = tl.float32, k = tl.float32, v = tl.float32, w = tl.float32,
      u = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_h = i_bh % $(c.H)
    p_q = q + i_bh * $(c.keyPitch) + i_k * $(c.BK) + tl.arange(0, $(c.BK)) + $(if c.reverse then (c.T-1)*c.K else 0)
    p_k = k + i_bh * $(c.keyPitch) + i_k * $(c.BK) + tl.arange(0, $(c.BK)) + $(if c.reverse then (c.T-1)*c.K else 0)
    p_v = v + i_bh * $(c.valuePitch) + i_v * $(c.BV) + tl.arange(0, $(c.BV)) + $(if c.reverse then (c.T-1)*c.V else 0)
    p_o = o + (i_bh + i_k * $(c.B) * $(c.H)) * ($(c.T) * $(c.V)) + i_v * $(c.BV) + tl.arange(0, $(c.BV)) + $(if c.reverse then (c.T-1)*c.V else 0)
    p_w = w + i_bh * $(c.keyPitch) + i_k * $(c.BK) + tl.arange(0, $(c.BK)) + $(if c.reverse then (c.T-1)*c.K else 0)
    p_u = u + i_h * $(c.K) + tl.arange(0, $(c.BK)) + i_k * $(c.BK)
    mask_bk = (i_k * $(c.BK) + tl.arange(0, $(c.BK))) < $(c.K)
    mask_bv = (i_v * $(c.BV) + tl.arange(0, $(c.BV))) < $(c.V)
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_h = tl.zeros([$(c.BV), $(c.BK)], dtype=tl.float32)
    if USE_INITIAL_STATE {
      p_h0 = h0 + i_bh * $(c.K) * $(c.V) +
        (i_k * $(c.BK) + tl.arange(0, $(c.BK))[None, :]) * $(c.V) +
        (i_v * $(c.BV) + tl.arange(0, $(c.BV))[:, None])
      b_h += tl.load(p_h0, mask=mask_kv, other=0.0).to(tl.float32)
    }
    b_u = tl.load(p_u, mask=mask_bk, other=0.0).to(tl.float32)
    for _i in range($(0), $(c.T), $(1)) {
      b_k = tl.load(p_k, mask=mask_bk, other=0.0).to(tl.float32)
      b_v = tl.load(p_v, mask=mask_bv, other=0.0).to(tl.float32)
      b_q = tl.load(p_q, mask=mask_bk, other=0.0).to(tl.float32) * scale
      b_w = tl.load(p_w, mask=mask_bk, other=0.0).to(tl.float32)
      b_w = tl.exp(b_w)
      b_kv = b_k[None, :] * b_v[:, None]
      b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
      b_o = tl.sum(b_o, axis=1)
      b_h = b_h * b_w[None, :]
      b_h += b_kv
      tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=mask_bv)
      if REVERSE {
        p_q -= $(c.K)
        p_k -= $(c.K)
        p_o -= $(c.V)
        p_v -= $(c.V)
        p_w -= $(c.K)
      } else {
        p_q += $(c.K)
        p_k += $(c.K)
        p_o += $(c.V)
        p_v += $(c.V)
        p_w += $(c.K)
      }
    }
    if STORE_FINAL_STATE {
      p_ht = ht + i_bh * $(c.K) * $(c.V) +
        (i_k * $(c.BK) + tl.arange(0, $(c.BK))[None, :]) * $(c.V) +
        (i_v * $(c.BV) + tl.arange(0, $(c.BV))[:, None])
      tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask_kv)
    }
  }

/-- The same complete source, with p_u/mask_bk/b_u moved just after i_h.
Issuing the independent U load before pointer setup and initial-state loading
allows its latency to overlap that prologue work; no performance claim is made. -/
def earlyUKernel (c : FusedRwkv6Kernel.Shape) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let w : RegionName := "w"
  let u : RegionName := "u"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  let USE_INITIAL_STATE := c.initial
  let STORE_FINAL_STATE := c.final
  let REVERSE := c.reverse
  triton {
    tl.param scale : tl.float32
    tl.region q = tl.float32, k = tl.float32, v = tl.float32, w = tl.float32,
      u = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_h = i_bh % $(c.H)
    p_u = u + i_h * $(c.K) + tl.arange(0, $(c.BK)) + i_k * $(c.BK)
    mask_bk = (i_k * $(c.BK) + tl.arange(0, $(c.BK))) < $(c.K)
    b_u = tl.load(p_u, mask=mask_bk, other=0.0).to(tl.float32)
    p_q = q + i_bh * $(c.keyPitch) + i_k * $(c.BK) + tl.arange(0, $(c.BK)) + $(if c.reverse then (c.T-1)*c.K else 0)
    p_k = k + i_bh * $(c.keyPitch) + i_k * $(c.BK) + tl.arange(0, $(c.BK)) + $(if c.reverse then (c.T-1)*c.K else 0)
    p_v = v + i_bh * $(c.valuePitch) + i_v * $(c.BV) + tl.arange(0, $(c.BV)) + $(if c.reverse then (c.T-1)*c.V else 0)
    p_o = o + (i_bh + i_k * $(c.B) * $(c.H)) * ($(c.T) * $(c.V)) + i_v * $(c.BV) + tl.arange(0, $(c.BV)) + $(if c.reverse then (c.T-1)*c.V else 0)
    p_w = w + i_bh * $(c.keyPitch) + i_k * $(c.BK) + tl.arange(0, $(c.BK)) + $(if c.reverse then (c.T-1)*c.K else 0)
    mask_bv = (i_v * $(c.BV) + tl.arange(0, $(c.BV))) < $(c.V)
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_h = tl.zeros([$(c.BV), $(c.BK)], dtype=tl.float32)
    if USE_INITIAL_STATE {
      p_h0 = h0 + i_bh * $(c.K) * $(c.V) +
        (i_k * $(c.BK) + tl.arange(0, $(c.BK))[None, :]) * $(c.V) +
        (i_v * $(c.BV) + tl.arange(0, $(c.BV))[:, None])
      b_h += tl.load(p_h0, mask=mask_kv, other=0.0).to(tl.float32)
    }
    for _i in range($(0), $(c.T), $(1)) {
      b_k = tl.load(p_k, mask=mask_bk, other=0.0).to(tl.float32)
      b_v = tl.load(p_v, mask=mask_bv, other=0.0).to(tl.float32)
      b_q = tl.load(p_q, mask=mask_bk, other=0.0).to(tl.float32) * scale
      b_w = tl.load(p_w, mask=mask_bk, other=0.0).to(tl.float32)
      b_w = tl.exp(b_w)
      b_kv = b_k[None, :] * b_v[:, None]
      b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
      b_o = tl.sum(b_o, axis=1)
      b_h = b_h * b_w[None, :]
      b_h += b_kv
      tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=mask_bv)
      if REVERSE {
        p_q -= $(c.K)
        p_k -= $(c.K)
        p_o -= $(c.V)
        p_v -= $(c.V)
        p_w -= $(c.K)
      } else {
        p_q += $(c.K)
        p_k += $(c.K)
        p_o += $(c.V)
        p_v += $(c.V)
        p_w += $(c.K)
      }
    }
    if STORE_FINAL_STATE {
      p_ht = ht + i_bh * $(c.K) * $(c.V) +
        (i_k * $(c.BK) + tl.arange(0, $(c.BK))[None, :]) * $(c.V) +
        (i_v * $(c.BV) + tl.arange(0, $(c.BV))[:, None])
      tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask_kv)
    }
  }
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
