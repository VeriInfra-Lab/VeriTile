import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.L2NormTriton1
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_N"]
  readPorts := ["X"]
  writePorts := ["Y"]

set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-! Complete selected FP32 sources. `tl.param` declares the runtime epsilon
register without emitting an assignment; the invocation supplies its actual
FP32 word. The candidate commutes only the two integer row-base products. -/

def template (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := B
  with_triton_call callSignature call_args [N := N] in
  triton {
    row = tl.program_id(0).to(tl.int64)
    X += row * N
    Y += row * N
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)
  }

def commutedTemplate (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := B
  with_triton_call callSignature call_args [N := N] in
  triton {
    row = tl.program_id(0).to(tl.int64)
    X += N * row
    Y += N * row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)
  }

def source (N B : Nat) : ComputeKernel :=
  template "X" "Y" N B

def indexCommuted (N B : Nat) : ComputeKernel :=
  commutedTemplate "X" "Y" N B

end VeriTile.Bench.Repairs.L2NormTriton1
