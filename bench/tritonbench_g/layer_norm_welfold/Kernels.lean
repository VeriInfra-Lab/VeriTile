/- Complete readable two-pass LayerNorm sources. The explicit ordered
FP32 sum trees retain the selected reduction topology; Y stores retain BF16. -/
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.WelfordSource
namespace VeriTile.Bench.Repairs.LayerNormWelfold.SourceCode
open VeriTile.Triton
noncomputable section

abbrev ReductionTree := DSL.WelfordSource.Tree
def blockCount (D B : Nat) : Nat := (D+B-1)/B
def rowKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : ComputeKernel := triton {
  tl.region X = tl.float32, W = tl.float32, Bias = tl.float32,
    Y = tl.bfloat16, Mean = tl.float32, Rstd = tl.float32
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(S)
  x0 = xindex
  tmp3 = tl.full([$(1), $(1)], 0, dtype=tl.float32)
  acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
  for k in tl.static_range($((D+B-1)/B)) {
    roffset = k * $(B)
    rindex = roffset + tl.arange(0, $(B))[None, :]
    rmask = rindex < $(D)
    x = tl.load(X + (rindex + ($(D) * x0)), mask=rmask, other=0.0).to(tl.float32)
    xx = (x).to(tl.float32)
    broadcasted = tl.broadcast_to(xx, [$(1), $(B)])
    next = acc + broadcasted
    acc = next
  }
  total = tl.sum(acc, axis=1, tree=$(tree), keep_dims=true)
  denominator = $(D)
  mu = total / denominator
  tl.debug_barrier()
  tl.store(Mean + x0, mu)
  tmp3 = mu
  acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
  for k in tl.static_range($((D+B-1)/B)) {
    roffset = k * $(B)
    rindex = roffset + tl.arange(0, $(B))[None, :]
    rmask = rindex < $(D)
    x = tl.load(X + (rindex + ($(D) * x0)), mask=rmask, other=0.0).to(tl.float32)
    xx = (x).to(tl.float32)
    delta = xx - tmp3
    square = delta * delta
    guarded = tl.where(rmask, square, 0.0)
    broadcasted = tl.broadcast_to(guarded, [$(1), $(B)])
    next = acc + broadcasted
    acc = next
  }
  total = tl.sum(acc, axis=1, tree=$(tree), keep_dims=true)
  denominator = $(D)
  variance = total / denominator
  epsilon = 1e-05
  inverse = tl.rsqrt(variance + epsilon)
  tl.debug_barrier()
  tl.store(Rstd + x0, inverse)
  tmp10 = inverse
  for k in tl.static_range($((D+B-1)/B)) {
    xrow = tl.reshape(x0, [])
    rowMean = tl.reshape(tmp3, [])
    rowInverse = tl.reshape(tmp10, [])
    roffset = k * $(B)
    rindex = roffset + tl.arange(0, $(B))
    rmask = rindex < $(D)
    tmp11 = tl.load(X + (rindex + ($(D) * xrow)), mask=rmask).to(tl.float32)
    tmp15 = tl.load(W + rindex, mask=rmask).to(tl.float32)
    tmp18 = tl.load(Bias + rindex, mask=rmask).to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - rowMean
    tmp14 = tmp13 * rowInverse
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    stored = (tmp21).to(tl.bfloat16)
    tl.store(Y + (rindex + ($(D) * xrow)), stored, mask=rmask)
  }
}

/-- Reuse the row offset for the X load and Y store in every normalization chunk. -/
def rowOffsetReusedKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : ComputeKernel := triton {
  tl.region X = tl.float32, W = tl.float32, Bias = tl.float32,
    Y = tl.bfloat16, Mean = tl.float32, Rstd = tl.float32
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(S)
  x0 = xindex
  tmp3 = tl.full([$(1), $(1)], 0, dtype=tl.float32)
  acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
  for k in tl.static_range($((D+B-1)/B)) {
    roffset = k * $(B)
    rindex = roffset + tl.arange(0, $(B))[None, :]
    rmask = rindex < $(D)
    x = tl.load(X + (rindex + ($(D) * x0)), mask=rmask, other=0.0).to(tl.float32)
    xx = (x).to(tl.float32)
    broadcasted = tl.broadcast_to(xx, [$(1), $(B)])
    next = acc + broadcasted
    acc = next
  }
  total = tl.sum(acc, axis=1, tree=$(tree), keep_dims=true)
  denominator = $(D)
  mu = total / denominator
  tl.debug_barrier()
  tl.store(Mean + x0, mu)
  tmp3 = mu
  acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
  for k in tl.static_range($((D+B-1)/B)) {
    roffset = k * $(B)
    rindex = roffset + tl.arange(0, $(B))[None, :]
    rmask = rindex < $(D)
    x = tl.load(X + (rindex + ($(D) * x0)), mask=rmask, other=0.0).to(tl.float32)
    xx = (x).to(tl.float32)
    delta = xx - tmp3
    square = delta * delta
    guarded = tl.where(rmask, square, 0.0)
    broadcasted = tl.broadcast_to(guarded, [$(1), $(B)])
    next = acc + broadcasted
    acc = next
  }
  total = tl.sum(acc, axis=1, tree=$(tree), keep_dims=true)
  denominator = $(D)
  variance = total / denominator
  epsilon = 1e-05
  inverse = tl.rsqrt(variance + epsilon)
  tl.debug_barrier()
  tl.store(Rstd + x0, inverse)
  tmp10 = inverse
  for k in tl.static_range($((D+B-1)/B)) {
    xrow = tl.reshape(x0, [])
    row_offset = $(D) * xrow
    rowMean = tl.reshape(tmp3, [])
    rowInverse = tl.reshape(tmp10, [])
    roffset = k * $(B)
    rindex = roffset + tl.arange(0, $(B))
    rmask = rindex < $(D)
    tmp11 = tl.load(X + (rindex + row_offset), mask=rmask).to(tl.float32)
    tmp15 = tl.load(W + rindex, mask=rmask).to(tl.float32)
    tmp18 = tl.load(Bias + rindex, mask=rmask).to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - rowMean
    tmp14 = tmp13 * rowInverse
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    stored = (tmp21).to(tl.bfloat16)
    tl.store(Y + (rindex + row_offset), stored, mask=rmask)
  }
}

/-- Select one of the two complete ordinary source bodies. -/
def selected (reused : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  if reused then rowOffsetReusedKernel X W Bias Y Mean Rstd S D B tree
  else rowKernel X W Bias Y Mean Rstd S D B tree

def runRows {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (kernel : ComputeKernel) :
    List Nat → FP.Structural.State α → Option (FP.Structural.State α)
  | [], s => some s
  | row :: rows, s => do
      let t ← FP.Structural.exec M kernel {s with pids := fun dim => if dim=0 then row else 0}
      runRows M kernel rows t

def launch {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (reused : Bool)
    (s : FP.Structural.State α) (Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : Option (FP.Structural.State α) :=
  runRows M (selected reused "X" "W" "Bias" Y Mean Rstd S D B tree) (List.range S) s

end
end VeriTile.Bench.Repairs.LayerNormWelfold.SourceCode
