import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.L2NormBwd
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

structure Strides where
  xRow : Nat
  xCol : Nat
  dyRow : Nat
  dyCol : Nat
  dxRow : Nat
  dxCol : Nat
  deriving DecidableEq, Repr

/-! Direct source definitions shared by Correct and FPEquiv. `tl.param`
declares the runtime scalar's type; the invocation supplies its FP32 value.
Both programs preserve the source casts, masks, reductions and arithmetic. -/

/-- Repaired production kernel, with independent X/DY/DX strides. -/
def kernel (X DY DX : RegionName) (s : Strides) (N B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X, DY], outputs=[DX]
  tl.region X = tl.float32, DY = tl.float32, DX = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  X += row * $(s.xRow)
  DX += row * $(s.dxRow)
  DY += row * $(s.dyRow)
  cols = tl.arange(0, $(B)).to(tl.int64)
  x = tl.load(X + cols * $(s.xCol), mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + eps)
  mask = cols < $(N)
  dy = tl.load(DY + cols * $(s.dyCol), mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
  tl.store(DX + cols * $(s.dxCol), dx, mask=mask)
}

/-- Existing candidate: commute only the three integer row/stride products. -/
def indexCommutedKernel (X DY DX : RegionName) (s : Strides) (N B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X, DY], outputs=[DX]
  tl.region X = tl.float32, DY = tl.float32, DX = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  X += $(s.xRow) * row
  DX += $(s.dxRow) * row
  DY += $(s.dyRow) * row
  cols = tl.arange(0, $(B)).to(tl.int64)
  x = tl.load(X + cols * $(s.xCol), mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + eps)
  mask = cols < $(N)
  dy = tl.load(DY + cols * $(s.dyCol), mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
  tl.store(DX + cols * $(s.dxCol), dx, mask=mask)
}

/- Public programs specialize the complete quotations, including their
region interface, runtime scalar declaration and every computation. -/
def source (s : Strides) (N B : Nat) : ComputeKernel :=
  kernel "X" "DY" "DX" s N B

def indexCommuted (s : Strides) (N B : Nat) : ComputeKernel :=
  indexCommutedKernel "X" "DY" "DX" s N B

end VeriTile.Bench.Repairs.L2NormBwd
