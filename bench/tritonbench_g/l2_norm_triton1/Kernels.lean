import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.L2NormTriton1
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-! Complete selected FP32 sources. `tl.param` declares the runtime epsilon
register without emitting an assignment; the invocation supplies its actual
FP32 word. The candidate commutes only the two integer row-base products. -/

def template (X Y : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  X += row * $(N)
  Y += row * $(N)
  cols = tl.arange(0, $(B)).to(tl.int64)
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + eps.to(tl.float32))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

def commutedTemplate (X Y : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  X += $(N) * row
  Y += $(N) * row
  cols = tl.arange(0, $(B)).to(tl.int64)
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + eps.to(tl.float32))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

/- Each complete quotation declares its public ports before pointer rebinding.
The selected entries transparently specialize that complete source. -/
def source (N B : Nat) : ComputeKernel :=
  template "X" "Y" N B

def indexCommuted (N B : Nat) : ComputeKernel :=
  commutedTemplate "X" "Y" N B

end VeriTile.Bench.Repairs.L2NormTriton1
