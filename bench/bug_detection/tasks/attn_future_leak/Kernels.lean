import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.attn_future_leak
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let q_ptr : RegionName := "q"
  let k_ptr : RegionName := "k"
  let v_ptr : RegionName := "v"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(16))
    h = tl.arange($(128))
    scale = tl.extra.cuda.libdevice.pow(128.0, -0.5).to(tl.float32)
    for head in range($(16)) {
      kv = head // $(1)
      q = tl.load($(q_ptr) + (head * $(16) + r[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
      k = tl.load($(k_ptr) + (kv * $(16) + j[None, :]) * $(128) + h[:, None], dtype=tl.bfloat16)
      s = tl.dot(q, k)
      s = s.to(tl.bfloat16)
      s = s.to(tl.float32) * scale
      blocked = j[None, :] > r[:, None]
      s = s + tl.where(blocked, -inf, 0.0)
      s = s.to(tl.float32)
      s = s - tl.max(s, axis=1, keep_dims=true)
      e = tl.exp(s)
      p = e / tl.sum(e, axis=1, keep_dims=true)
      p = p.to(tl.bfloat16)
      v = tl.load($(v_ptr) + (kv * $(16) + j[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
      y = tl.dot(p, v).to(tl.bfloat16)
      tl.store($(out_ptr) + (r[:, None] * $(16) + head) * $(128) + h[None, :], y)
    }
  }

def candidate : ComputeKernel :=
  let q_ptr : RegionName := "q"
  let k_ptr : RegionName := "k"
  let v_ptr : RegionName := "v"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(16))
    h = tl.arange($(128))
    scale = tl.extra.cuda.libdevice.pow(128.0, -0.5).to(tl.float32)
    for head in range($(16)) {
      kv = head // $(1)
      q = tl.load($(q_ptr) + (head * $(16) + r[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
      k = tl.load($(k_ptr) + (kv * $(16) + j[None, :]) * $(128) + h[:, None], dtype=tl.bfloat16)
      s = tl.dot(q, k)
      s = s.to(tl.bfloat16)
      s = s.to(tl.float32) * scale
      blocked = j[None, :] > r[:, None] + $(1)
      s = s + tl.where(blocked, -inf, 0.0)
      s = s.to(tl.float32)
      s = s - tl.max(s, axis=1, keep_dims=true)
      e = tl.exp(s)
      p = e / tl.sum(e, axis=1, keep_dims=true)
      p = p.to(tl.bfloat16)
      v = tl.load($(v_ptr) + (kv * $(16) + j[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
      y = tl.dot(p, v).to(tl.bfloat16)
      tl.store($(out_ptr) + (r[:, None] * $(16) + head) * $(128) + h[None, :], y)
    }
  }

end Kernels
end
end VeriTile.Bench.BugDetection.attn_future_leak
