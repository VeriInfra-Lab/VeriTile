import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_row_parallel_tp2
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let out_ptr : RegionName := "out"
  let partial_ptr : RegionName := "partial"
  triton {
    r = tl.arange($(16))
    k = tl.arange($(2048))
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + (k)[None, :], dtype=tl.bfloat16)
    w = tl.load($(weight_ptr) + (j)[None, :] * $(2048) + (k)[:, None], dtype=tl.bfloat16)
    y = tl.dot(x, w).to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + (j)[None, :], y)
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let out_ptr : RegionName := "out"
  let partial_ptr : RegionName := "partial"
  triton {
    r = tl.arange($(16))
    k = tl.arange($(1024))
    j = tl.arange($(2048))
    for shard in range($(2)) {
      x = tl.load($(x_ptr) + r[:, None] * $(2048) + (k + shard * $(1024))[None, :], dtype=tl.bfloat16)
      w = tl.load($(weight_ptr) + (j)[None, :] * $(2048) + (k + shard * $(1024))[:, None], dtype=tl.bfloat16)
      y = tl.dot(x, w).to(tl.bfloat16)
      if shard == $(0) {
        tl.store($(partial_ptr) + r[:, None] * $(2048) + j[None, :], y)
      } else {
        shard_value = tl.load($(partial_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
        y = (shard_value + y).to(tl.bfloat16)
        tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y)
      }
    }
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_row_parallel_tp2
