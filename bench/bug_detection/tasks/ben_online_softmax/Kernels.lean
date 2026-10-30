import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_online_softmax
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
    r = tl.arange($(64))
    j = tl.arange($(64))
    h = tl.arange($(128))
    scale = tl.extra.cuda.libdevice.pow(128.0, -0.5).to(tl.float32)
    for head in range($(16)) {
      kv = head // $(1)
      q = tl.load($(q_ptr) + (head * $(64) + r[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
      k = tl.load($(k_ptr) + (kv * $(64) + j[None, :]) * $(128) + h[:, None], dtype=tl.bfloat16)
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
      v = tl.load($(v_ptr) + (kv * $(64) + j[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
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
    r = tl.arange($(64))
    j = tl.arange($(64))
    h = tl.arange($(128))
    scale = tl.extra.cuda.libdevice.pow(128.0, -0.5).to(tl.float32)
    for head in range($(16)) {
      kv = head // $(1)
      q = tl.load($(q_ptr) + (head * $(64) + r[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
      m = tl.full([$(64), $(1)], -inf, dtype=tl.float32)
      l = tl.zeros([$(64), $(1)], dtype=tl.float32)
      acc = tl.zeros([$(64), $(128)], dtype=tl.float32)
      kaxis = tl.arange($(16))
      for block in range($(4)) {
        key = block * $(16) + kaxis
        k = tl.load($(k_ptr) + (kv * $(64) + key[None, :]) * $(128) + h[:, None], dtype=tl.bfloat16)
        v = tl.load($(v_ptr) + (kv * $(64) + key[:, None]) * $(128) + h[None, :], dtype=tl.bfloat16)
        v = v.to(tl.float32)
        s = tl.dot(q, k).to(tl.bfloat16).to(tl.float32) * scale
        s = s + tl.where(key[None, :] > r[:, None], -inf, 0.0)
        newm = tl.maximum(m, tl.max(s, axis=1, keep_dims=true))
        alpha = tl.exp(m - newm)
        p = tl.exp(s - newm)
        l = l * alpha + tl.sum(p, axis=1, keep_dims=true)
        acc = acc * alpha + tl.dot(p, v)
        m = newm
      }
      y = (acc / l).to(tl.bfloat16)
      tl.store($(out_ptr) + (r[:, None] * $(16) + head) * $(128) + h[None, :], y)
    }
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_online_softmax
