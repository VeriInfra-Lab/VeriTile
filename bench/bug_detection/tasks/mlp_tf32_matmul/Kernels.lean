import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.mlp_tf32_matmul
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
    k = tl.arange($(2048))
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + (k)[None, :], dtype=tl.float32)
    w = tl.load($(weight_ptr) + (j)[None, :] * $(2048) + (k)[:, None], dtype=tl.float32)
    y = tl.dot(x, w, input_precision="ieee").to(tl.float32)
    tl.store($(out_ptr) + r[:, None] * $(2048) + (j)[None, :], y)
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(64))
    k = tl.arange($(2048))
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + (k)[None, :], dtype=tl.float32)
    w = tl.load($(weight_ptr) + (j)[None, :] * $(2048) + (k)[:, None], dtype=tl.float32)
    y = tl.dot(x, w, input_precision="tf32").to(tl.float32)
    tl.store($(out_ptr) + r[:, None] * $(2048) + (j)[None, :], y)
  }

end Kernels
end
end VeriTile.Bench.BugDetection.mlp_tf32_matmul
