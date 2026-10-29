import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_embedding_onehot
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(64))
    j = tl.arange($(1024))
    h = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(1024) + j[None, :], dtype=tl.bfloat16)
    index = tl.argmax(x, axis=1)
    y = tl.load($(weight_ptr) + index[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + h[None, :], y)
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(64))
    j = tl.arange($(1024))
    h = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(1024) + j[None, :], dtype=tl.bfloat16)
    index = tl.argmax(x, axis=1)
    indicator = tl.where(j[None, :] == index[:, None], 1.0, 0.0).to(tl.bfloat16)
    w = tl.load($(weight_ptr) + j[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    y = tl.dot(indicator, w).to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + h[None, :], y)
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_embedding_onehot
