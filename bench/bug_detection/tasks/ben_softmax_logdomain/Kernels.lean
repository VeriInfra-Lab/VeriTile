import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_softmax_logdomain
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(256))
    x = tl.load($(x_ptr) + r[:, None] * $(256) + j[None, :], dtype=tl.bfloat16)
    f = x.to(tl.float32)
    m = tl.max(f, axis=1, keep_dims=true)
    z = f - m
    e = tl.exp(z)
    s = tl.sum(e, axis=1, keep_dims=true)
    y = e / s
    tl.store($(out_ptr) + r[:, None] * $(256) + j[None, :], y.to(tl.bfloat16))
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(256))
    x = tl.load($(x_ptr) + r[:, None] * $(256) + j[None, :], dtype=tl.bfloat16)
    f = x.to(tl.float32)
    m = tl.max(f, axis=1, keep_dims=true)
    z = f - m
    e = tl.exp(z)
    s = tl.sum(e, axis=1, keep_dims=true)
    y = tl.exp(z - tl.log(s))
    tl.store($(out_ptr) + r[:, None] * $(256) + j[None, :], y.to(tl.bfloat16))
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_softmax_logdomain
