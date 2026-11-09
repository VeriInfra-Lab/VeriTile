import VeriTile.Triton.Source
import VeriTile.Triton.DSL.ConstexprFloatExpressions

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
  compiled : ConstexprFloat.Model

/-- The source constexpr tree, interpreted primitive by primitive before
runtime tensor execution. The Real interpretation is used only by Correct. -/
def factorTree (channels : Nat) : ConstexprFloat.Expr :=
  .mul (.pow (.natural channels) (.neg (.literal 0.5))) (.literal 1.44269504)

def Parameters.factor (p : Parameters) : ℝ := if p.query then ConstexprFloat.eval p.compiled (factorTree p.channels) else 1

/-- Select exact real constexpr primitives for the independent mathematical proof. -/
def Parameters.mathematical (p : Parameters) : Parameters := {p with compiled := ConstexprFloat.real}

def signature : CallSignature where
  pointers := [("X",.fp32),("X_int8",.i8),("Scale",.fp32)]
  scalars := []
  constexpr := ["L","C","BLK","BLOCKS","BASE","WIDTH","IS_Q"]
  readPorts := ["X"]
  writePorts := ["X_int8","Scale"]

/-- Complete repaired Python function body; invocation metadata stays outside. -/
def sourceKernel (p : Parameters) : ComputeKernel :=
  let X := xRegion
  let X_int8 := outRegion
  let Scale := scaleRegion
  let L := p.length
  let C := p.channels
  let BLK := p.blockRows
  let BLOCKS := p.blocks
  let BASE := p.base
  let WIDTH := p.width
  let IS_Q := p.query
  with_constexpr_float_model p.compiled in
  with_triton_call signature call_args [] in triton {
    block = tl.program_id(0).to(tl.int64) + BASE
    batch = block // BLOCKS
    local_block = block % BLOCKS
    lane = tl.arange(0, WIDTH).to(tl.int64)
    row = local_block * BLK + lane // C
    offset = batch * L * C + local_block * BLK * C + lane
    active = (lane < BLK * C) & (row < L)
    x = tl.load(X + offset, mask=active, other=0.0)
    if IS_Q:
        x = x * (C**-0.5 * 1.44269504)
    finite = tl.abs(x) <= 3.4028234663852886e38
    bad = tl.max(tl.where(finite, 0, 1), 0)
    x = tl.where(finite, x, 0.0)
    maximum = tl.max(tl.abs(x), 0)
    scale = maximum / 127.0
    usable = (bad == 0) & (scale > 0.0)
    denominator = tl.where(usable, scale, 1.0)
    normalized = x / denominator
    normalized = tl.where(usable, normalized, 0.0)
    normalized = tl.minimum(tl.maximum(normalized, -127.0), 127.0)
    biased = normalized + 0.5 * tl.where(normalized >= 0.0, 1.0, -1.0)
    quantized = biased.to(tl.int8)
    tl.store(X_int8 + offset, quantized, mask=active)
    tl.store(Scale + block, tl.where(bad == 0, scale, 0.0))
  }

/-- Complete repaired Python function body; invocation metadata stays outside. -/
def biasFoldKernel (p : Parameters) : ComputeKernel :=
  let X := xRegion
  let X_int8 := outRegion
  let Scale := scaleRegion
  let L := p.length
  let C := p.channels
  let BLK := p.blockRows
  let BLOCKS := p.blocks
  let BASE := p.base
  let WIDTH := p.width
  let IS_Q := p.query
  with_constexpr_float_model p.compiled in
  with_triton_call signature call_args [] in triton {
    block = tl.program_id(0).to(tl.int64) + BASE
    batch = block // BLOCKS
    local_block = block % BLOCKS
    lane = tl.arange(0, WIDTH).to(tl.int64)
    row = local_block * BLK + lane // C
    offset = batch * L * C + local_block * BLK * C + lane
    active = (lane < BLK * C) & (row < L)
    x = tl.load(X + offset, mask=active, other=0.0)
    if IS_Q:
        x = x * (C**-0.5 * 1.44269504)
    finite = tl.abs(x) <= 3.4028234663852886e38
    bad = tl.max(tl.where(finite, 0, 1), 0)
    x = tl.where(finite, x, 0.0)
    maximum = tl.max(tl.abs(x), 0)
    scale = maximum / 127.0
    usable = (bad == 0) & (scale > 0.0)
    denominator = tl.where(usable, scale, 1.0)
    normalized = x / denominator
    normalized = tl.where(usable, normalized, 0.0)
    normalized = tl.minimum(tl.maximum(normalized, -127.0), 127.0)
    biased = normalized + tl.where(normalized >= 0.0, 0.5, -0.5)
    quantized = biased.to(tl.int8)
    tl.store(X_int8 + offset, quantized, mask=active)
    tl.store(Scale + block, tl.where(bad == 0, scale, 0.0))
  }


end
end VeriTile.Bench.Repairs.Int8Quantization
