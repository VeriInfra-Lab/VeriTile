import VeriTile.Triton
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.RmsnormFused
open VeriTile.Triton

/-- Complete repaired runtime-epsilon source. The FP32 scalar is supplied by
`Runtime.bind` and the IO invocation; its declaration emits no statement.
The int64 PID conversion, both loops and undefined inactive W are retained. -/
def source (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  Y += row * $(N)
  X += row * $(N)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + eps)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}

/-- Compatibility name for the complete readable runtime source. -/
abbrev template (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  source X Y W N B

/-- The existing output-address candidate, with only `row*N` changed to
`N*row`. Both complete numerical loops and the runtime FP32 scalar remain. -/
def addressCommutedSource (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  Y += $(N) * row
  X += row * $(N)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + eps)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}

structure Shape where
  leading : List Nat
  width : Nat
  block : Nat

def Shape.rows (s : Shape) : Nat := s.leading.prod
def Shape.dims (s : Shape) : List Nat := s.leading ++ [s.width]
def Shape.Valid (s : Shape) : Prop :=
  0 < s.width ∧ s.width ≤ s.block ∧ s.block ≤ 16384 ∧
  s.rows * s.width ≤ (2^63-1)/4

def Shape.offset (s : Shape) (row col : Nat) : Nat := row*s.width+col
def Shape.grid (s : Shape) : List Nat := List.range s.rows

end VeriTile.Bench.Repairs.RmsnormFused
