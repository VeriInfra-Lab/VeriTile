import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_gelu_handrolled
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
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
    f = x.to(tl.float32)
    a = 0.5 * f * (1.0 + tl.tanh(0.7978845608028654 * (f + 0.044715 * f * f * f)))
    y = a
    tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y.to(tl.bfloat16))
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
    f = x.to(tl.float32)
    a = 0.5 * f * (1.0 + tl.tanh(0.7978845608028654 * (f + 0.044715 * f * f * f)))
    y = a
    tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y.to(tl.bfloat16))
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_gelu_handrolled
