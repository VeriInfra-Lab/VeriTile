import bench.repairs.tritonbench_g.l2_norm_triton1.Kernels

namespace VeriTile.Bench.Repairs.L2NormTriton2
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- The two Python forward bodies are identical. -/
abbrev forward := L2NormTriton1.source
abbrev forwardCommuted := L2NormTriton1.indexCommuted

/-! Complete FP32 backward sources. Epsilon is a raw FP32 ABI input declared
with `tl.param`. Both reductions and the final expression retain their source
order; the candidate changes only the three integer row products. -/

def backwardTemplate (X DY DX : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X, DY], outputs=[DX]
  tl.region X = tl.float32, DY = tl.float32, DX = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  X += row * $(N)
  DX += row * $(N)
  DY += row * $(N)
  cols = tl.arange(0, $(B)).to(tl.int64)
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + eps.to(tl.float32))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps.to(tl.float32))) * rstd * x
  tl.store(DX + cols, dx, mask=mask)
}

def commutedBackwardTemplate (X DY DX : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X, DY], outputs=[DX]
  tl.region X = tl.float32, DY = tl.float32, DX = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  X += $(N) * row
  DX += $(N) * row
  DY += $(N) * row
  cols = tl.arange(0, $(B)).to(tl.int64)
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + eps.to(tl.float32))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps.to(tl.float32))) * rstd * x
  tl.store(DX + cols, dx, mask=mask)
}

/- Specialize the complete templates, including their declared ports. -/
def backward (N B : Nat) : ComputeKernel :=
  backwardTemplate "X" "DY" "DX" N B

def backwardCommuted (N B : Nat) : ComputeKernel :=
  commutedBackwardTemplate "X" "DY" "DX" N B

end VeriTile.Bench.Repairs.L2NormTriton2
