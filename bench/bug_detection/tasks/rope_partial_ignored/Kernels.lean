import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.rope_partial_ignored
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let q_ptr : RegionName := "q"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(256))
    j = tl.arange($(128))
    position = r % $(16)
    channel = j % $(16)
    frequency = tl.extra.cuda.libdevice.pow(10000.0, -2.0 * tl.toReal(channel).to(tl.float64) / 32.0)
    frequency = frequency.to(tl.float32)
    angle = (tl.toReal(position[:, None] + $(0)).to(tl.float32) * frequency[None, :]).to(tl.float32).to(tl.float32)
    co = tl.cos(angle).to(tl.bfloat16)
    si = tl.sin(angle).to(tl.bfloat16)
    x = tl.load($(q_ptr) + r[:, None] * $(128) + j[None, :], dtype=tl.bfloat16)
    x = x.to(tl.bfloat16)
    pair = tl.where(j < $(16), j + $(16), j - $(16))
    z = tl.load($(q_ptr) + r[:, None] * $(128) + pair[None, :], dtype=tl.bfloat16)
    z = z.to(tl.bfloat16)
    z = tl.where(j < $(16), -z, z)
    y = x * co + z * si
    y = y.to(tl.bfloat16)
    original = tl.load($(q_ptr) + r[:, None] * $(128) + j[None, :], dtype=tl.bfloat16)
    y = tl.where(j[None, :] < $(32), y, original)
    tl.store($(out_ptr) + r[:, None] * $(128) + j[None, :], y)
  }

def candidate : ComputeKernel :=
  let q_ptr : RegionName := "q"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(256))
    j = tl.arange($(128))
    position = r % $(16)
    channel = j % $(64)
    frequency = tl.extra.cuda.libdevice.pow(10000.0, -2.0 * tl.toReal(channel).to(tl.float64) / 128.0)
    frequency = frequency.to(tl.float32)
    angle = (tl.toReal(position[:, None] + $(0)).to(tl.float32) * frequency[None, :]).to(tl.float32).to(tl.float32)
    co = tl.cos(angle).to(tl.bfloat16)
    si = tl.sin(angle).to(tl.bfloat16)
    x = tl.load($(q_ptr) + r[:, None] * $(128) + j[None, :], dtype=tl.bfloat16)
    x = x.to(tl.bfloat16)
    pair = tl.where(j < $(64), j + $(64), j - $(64))
    z = tl.load($(q_ptr) + r[:, None] * $(128) + pair[None, :], dtype=tl.bfloat16)
    z = z.to(tl.bfloat16)
    z = tl.where(j < $(64), -z, z)
    y = x * co + z * si
    y = y.to(tl.bfloat16)
    original = tl.load($(q_ptr) + r[:, None] * $(128) + j[None, :], dtype=tl.bfloat16)
    y = tl.where(j[None, :] < $(128), y, original)
    tl.store($(out_ptr) + r[:, None] * $(128) + j[None, :], y)
  }

end Kernels
end
end VeriTile.Bench.BugDetection.rope_partial_ignored
