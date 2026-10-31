import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure

/-! Complete selected FP32 streaming source. Both ordinary quotations retain
all casts, accumulation, reduction, division, masks and explicit source ports.
Both interpretations use these same repaired FP32 sources. -/
namespace VeriTile.Bench.Repairs.MeanReduction
open VeriTile.Triton

/-- Original source, before the existing integer-index transformation. -/
def sourceKernel (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X], outputs=[Mean]
  tl.region X = tl.float32, Mean = tl.float32
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

/-- Existing candidate: commute only the natural program-base product. -/
def indexCommutedSource (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X], outputs=[Mean]
  tl.region X = tl.float32, Mean = tl.float32
  pid = $(BLOCK_M) * tl.program_id(0) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

def gridSize (rows : Nat) : Nat := (rows+7)/8

def kernel (M N : Nat) : ComputeKernel := sourceKernel "X" "Mean" M N 8 8

def indexCommutedKernel (M N : Nat) : ComputeKernel := indexCommutedSource "X" "Mean" M N 8 8

end VeriTile.Bench.Repairs.MeanReduction
