import bench.repairs.tritonbench_g.l2_norm_triton1.Kernels

/-! Forward source/candidate bodies are literally shared with l2_norm_triton1.
The complete backward Python functions are pasted below. Runtime epsilon is
an FP32 scalar input; N is a runtime integer and only BLOCK_N is constexpr. -/
namespace VeriTile.Bench.Repairs.L2NormTriton2
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

abbrev forward := L2NormTriton1.source
abbrev forwardCommuted := L2NormTriton1.indexCommuted

def backwardSignature : CallSignature where
  pointers := [("X",.fp32),("DY",.fp32),("DX",.fp32)]
  scalars := [("N",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_N"]
  readPorts := ["X","DY"]
  writePorts := ["DX"]

def backwardTemplate (X DY DX : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := B
  with_triton_call backwardSignature call_args [N := N] in
  triton {
    row = tl.program_id(0).to(tl.int64)
    X += row * N
    DX += row * N
    DY += row * N
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)
  }

def commutedBackwardTemplate (X DY DX : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := B
  with_triton_call backwardSignature call_args [N := N] in
  triton {
    row = tl.program_id(0).to(tl.int64)
    X += N * row
    DX += N * row
    DY += N * row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)
  }

def backward (N B : Nat) : ComputeKernel :=
  backwardTemplate "X" "DY" "DX" N B

def backwardCommuted (N B : Nat) : ComputeKernel :=
  commutedBackwardTemplate "X" "DY" "DX" N B

end VeriTile.Bench.Repairs.L2NormTriton2
