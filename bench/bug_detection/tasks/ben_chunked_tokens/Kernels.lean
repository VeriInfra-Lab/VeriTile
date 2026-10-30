import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_chunked_tokens
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let gate_ptr : RegionName := "gate"
  let up_ptr : RegionName := "up"
  let down_ptr : RegionName := "down"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    h = tl.arange($(2048))
    j = tl.arange($(1024))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    wg = tl.load($(gate_ptr) + j[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    wu = tl.load($(up_ptr) + j[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    g = tl.dot(x, wg).to(tl.bfloat16)
    u = tl.dot(x, wu).to(tl.bfloat16)
    a = silu(g)
    a = a.to(tl.bfloat16)
    a = (a * u).to(tl.bfloat16)
    wd = tl.load($(down_ptr) + h[None, :] * $(1024) + j[:, None], dtype=tl.bfloat16)
    y = tl.dot(a, wd).to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + h[None, :], y)
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let gate_ptr : RegionName := "gate"
  let up_ptr : RegionName := "up"
  let down_ptr : RegionName := "down"
  let out_ptr : RegionName := "out"
  triton {
    for chunk in range($(2)) {
      r = tl.arange($(8))
      h = tl.arange($(2048))
      j = tl.arange($(1024))
      x = tl.load($(x_ptr) + (r[:, None] + chunk * $(8)) * $(2048) + h[None, :], dtype=tl.bfloat16)
      wg = tl.load($(gate_ptr) + j[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
      wu = tl.load($(up_ptr) + j[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
      g = tl.dot(x, wg).to(tl.bfloat16)
      u = tl.dot(x, wu).to(tl.bfloat16)
      a = silu(g)
      a = a.to(tl.bfloat16)
      a = (a * u).to(tl.bfloat16)
      wd = tl.load($(down_ptr) + h[None, :] * $(1024) + j[:, None], dtype=tl.bfloat16)
      y = tl.dot(a, wd).to(tl.bfloat16)
      tl.store($(out_ptr) + (r[:, None] + chunk * $(8)) * $(2048) + h[None, :], y)
    }
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_chunked_tokens
