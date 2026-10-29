import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.swiglu__default
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
    j = tl.arange($(6144))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    wg = tl.load($(gate_ptr) + j[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    wu = tl.load($(up_ptr) + j[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    g = tl.dot(x, wg).to(tl.bfloat16)
    u = tl.dot(x, wu).to(tl.bfloat16)
    a = silu(g)
    a = a.to(tl.bfloat16)
    a = (a * u).to(tl.bfloat16)
    wd = tl.load($(down_ptr) + h[None, :] * $(6144) + j[:, None], dtype=tl.bfloat16)
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
    r = tl.arange($(16))
    h = tl.arange($(2048))
    j = tl.arange($(6144))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    p = tl.arange($(12288))
    wg = tl.load($(gate_ptr) + p[None, :] * $(2048) + h[:, None], mask=p[None, :] < $(6144), other=0, dtype=tl.bfloat16)
    wu = tl.load($(up_ptr) + (p[None, :] - $(6144)) * $(2048) + h[:, None], mask=p[None, :] >= $(6144), other=0, dtype=tl.bfloat16)
    wp = tl.where(p[None, :] < $(6144), wg, wu)
    gu = tl.dot(x, wp).to(tl.bfloat16)
    gu = tl.trans(tl.reshape(gu, [$(16), $(2), $(6144)]))
    g = tl.split(gu, 0)
    u = tl.split(gu, 1)
    a = silu(g)
    a = a.to(tl.bfloat16)
    a = (a * u).to(tl.bfloat16)
    wd = tl.load($(down_ptr) + h[None, :] * $(6144) + j[:, None], dtype=tl.bfloat16)
    y = tl.dot(a, wd).to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + h[None, :], y)
  }

end Kernels
end
end VeriTile.Bench.BugDetection.swiglu__default
