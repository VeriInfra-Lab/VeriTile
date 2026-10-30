import VeriTile.Triton.Source
import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.Int8Quantization
open VeriTile.Triton
noncomputable section

def xRegion : RegionName := "x"
def outRegion : RegionName := "out"
def scaleRegion : RegionName := "scale"

structure Parameters where
  batches : Nat
  length : Nat
  channels : Nat
  blockRows : Nat
  blocks : Nat
  base : Nat
  width : Nat
  query : Bool
  factor : ℝ

/-- Full floating/INT8 source. The natural address projection retains all
operations; `Direct.native_pointers` proves equality with the widened int64
address and mask operations for every admitted launch. `factor`
is the host-computed Q constant after its FP32 literal conversion. K performs
no multiplication. No supplied-scale or pre-rounding slice is substituted. -/
def source (p : Parameters) : ComputeKernel := triton {
  tl.region xRegion = tl.float32, outRegion = tl.int8, scaleRegion = tl.float32
  block = tl.program_id(0) + $(p.base)
  batch = block // $(p.blocks)
  local_block = block % $(p.blocks)
  lane = tl.arange(0, $(p.width))
  row = local_block * $(p.blockRows) + lane // $(p.channels)
  offset = batch * $(p.length) * $(p.channels) + local_block * $(p.blockRows) * $(p.channels) + lane
  active = (lane < $(p.blockRows * p.channels)) & (row < $(p.length))
  x = tl.load(xRegion + offset, mask=active, other=0.0)
  if $((p.query : Bool)) { x = x * $((p.factor : ℝ)) }
  finite = tl.abs(x) <= 340282346638528859811704183484516925440.0
  bad = tl.max(tl.where(finite, $((0 : Int)), $((1 : Int))), 0)
  x = tl.where(finite, x, 0.0)
  maximum = tl.max(tl.abs(x), 0)
  scale = maximum / 127.0
  usable = (bad == $((0 : Int))) & (scale > 0.0)
  denominator = tl.where(usable, scale, 1.0)
  normalized = x / denominator
  normalized = tl.where(usable, normalized, 0.0)
  normalized = tl.minimum(tl.maximum(normalized, -127.0), 127.0)
  biased = normalized + 0.5 * tl.where(normalized >= 0.0, 1.0, -1.0)
  quantized = (biased).to(tl.int8)
  tl.store(outRegion + offset, quantized, mask=active)
  tl.store(scaleRegion + block, tl.where(bad == $((0 : Int)), scale, 0.0))
}

/-- Eliminate the per-lane multiplication used to select the half-unit bias.
All classification, reductions, casts and stores remain in source order. -/
def biasFoldSource (p : Parameters) : ComputeKernel := triton {
  tl.region xRegion = tl.float32, outRegion = tl.int8, scaleRegion = tl.float32
  block = tl.program_id(0) + $(p.base)
  batch = block // $(p.blocks)
  local_block = block % $(p.blocks)
  lane = tl.arange(0, $(p.width))
  row = local_block * $(p.blockRows) + lane // $(p.channels)
  offset = batch * $(p.length) * $(p.channels) + local_block * $(p.blockRows) * $(p.channels) + lane
  active = (lane < $(p.blockRows * p.channels)) & (row < $(p.length))
  x = tl.load(xRegion + offset, mask=active, other=0.0)
  if $((p.query : Bool)) { x = x * $((p.factor : ℝ)) }
  finite = tl.abs(x) <= 340282346638528859811704183484516925440.0
  bad = tl.max(tl.where(finite, $((0 : Int)), $((1 : Int))), 0)
  x = tl.where(finite, x, 0.0)
  maximum = tl.max(tl.abs(x), 0)
  scale = maximum / 127.0
  usable = (bad == $((0 : Int))) & (scale > 0.0)
  denominator = tl.where(usable, scale, 1.0)
  normalized = x / denominator
  normalized = tl.where(usable, normalized, 0.0)
  normalized = tl.minimum(tl.maximum(normalized, -127.0), 127.0)
  biased = normalized + tl.where(normalized >= 0.0, 0.5, -0.5)
  quantized = (biased).to(tl.int8)
  tl.store(outRegion + offset, quantized, mask=active)
  tl.store(scaleRegion + block, tl.where(bad == $((0 : Int)), scale, 0.0))
}

def originalSHA256 : String := "d1bfce576367aee5dec15ea8da413cc2eabc7a1a30363718bacfa7167bb8e34d"
def historicalCandidateSHA256 : String := "011d56e3dab83a0b4f6a9fc862628edd527ea3bfdac57dc1c8b4497587653c02"

end
end VeriTile.Bench.Repairs.Int8Quantization
