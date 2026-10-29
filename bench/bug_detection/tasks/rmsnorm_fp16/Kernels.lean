import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.rmsnorm_fp16
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
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
    w = tl.load($(weight_ptr) + j, dtype=tl.float32)
    f = x.to(tl.float32)
    sq = f * f
    ss = tl.sum(sq, axis=1, keep_dims=true)
    variance = ss / 2048.0
    variance = variance + 0.000001
    z = f * tl.rsqrt(variance)
    z = z.to(tl.bfloat16)
    y = w * z
    tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y)
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(64))
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
    w = tl.load($(weight_ptr) + j, dtype=tl.float32)
    f = x.to(tl.float16)
    sq = f * f
    ss = tl.sum(sq, axis=1, keep_dims=true, dtype=tl.float16)
    variance = ss / 2048.0
    variance = variance.to(tl.float16)
    eps = tl.full([], 0.000001, dtype=tl.float16)
    variance = (variance + eps).to(tl.float16)
    inv = tl.rsqrt(variance).to(tl.float16)
    z = (f * inv).to(tl.float16)
    w = w.to(tl.float16)
    y = w * z
    y = y.to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y)
  }

end Kernels
end
end VeriTile.Bench.BugDetection.rmsnorm_fp16
