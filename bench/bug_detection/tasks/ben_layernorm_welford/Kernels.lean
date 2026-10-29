import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_layernorm_welford
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let bias_ptr : RegionName := "bias"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(2048))
    x = tl.load($(x_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
    w = tl.load($(weight_ptr) + j, dtype=tl.float32)
    f = x.to(tl.float32)
    mean = tl.sum(f, axis=1, keep_dims=true) / $(2048)
    f = f - mean
    sq = f * f
    ss = tl.sum(sq, axis=1, keep_dims=true)
    variance = ss / 2048.0
    variance = variance + 0.000001
    z = f * tl.rsqrt(variance)
    y = w * z
    bias = tl.load($(bias_ptr) + j, dtype=tl.float32)
    y = y + bias
    y = y.to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y)
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let weight_ptr : RegionName := "weight"
  let bias_ptr : RegionName := "bias"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(16))
    j = tl.arange($(2048))
    w = tl.load($(weight_ptr) + j, dtype=tl.float32)
    k = tl.arange($(256))
    v = tl.load($(x_ptr) + r[:, None] * $(2048) + k[None, :], dtype=tl.bfloat16)
    v = v.to(tl.float32)
    mean = tl.sum(v, axis=1, keep_dims=true) / $(256)
    delta = v - mean
    m2 = tl.sum(delta * delta, axis=1, keep_dims=true)
    count = $(256)
    for chunk in range($(1), $(8)) {
      v = tl.load($(x_ptr) + r[:, None] * $(2048) + chunk * $(256) + k[None, :], dtype=tl.bfloat16)
      v = v.to(tl.float32)
      mc = tl.sum(v, axis=1, keep_dims=true) / $(256)
      dc = v - mc
      m2c = tl.sum(dc * dc, axis=1, keep_dims=true)
      delta = mc - mean
      total = count + $(256)
      mean = mean + delta * (tl.toReal($(256)) / tl.toReal(total))
      m2 = m2 + m2c + delta * delta * (tl.toReal(count * $(256)) / tl.toReal(total))
      count = total
    }
    f = tl.load($(x_ptr) + r[:, None] * $(2048) + j[None, :], dtype=tl.bfloat16)
    f = f.to(tl.float32)
    f = f - mean
    variance = m2 / tl.toReal(count)
    variance = variance + 0.000001
    z = f * tl.rsqrt(variance)
    y = w * z
    bias = tl.load($(bias_ptr) + j, dtype=tl.float32)
    y = y + bias
    y = y.to(tl.bfloat16)
    tl.store($(out_ptr) + r[:, None] * $(2048) + j[None, :], y)
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_layernorm_welford
