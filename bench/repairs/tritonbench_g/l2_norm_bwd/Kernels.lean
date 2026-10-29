import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Complete selected original/candidate Python functions. All signed-int64
runtime stride/count arguments are bound outside the body, epsilon remains a
runtime FP32 argument, and only BLOCK_N is constexpr. The public Nat stride
geometry and all prior IO/native scopes are retained. -/
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

def sourceSignature : CallSignature where
  pointers := [("X",.fp32),("DY",.fp32),("DX",.fp32)]
  scalars := [("stride_x_row",.i64),("stride_x_col",.i64),
    ("stride_dy_row",.i64),("stride_dy_col",.i64),("stride_dx_row",.i64),
    ("stride_dx_col",.i64),("N",.i64),("eps",.fp32)]
  constexpr := ["BLOCK_N"]
  readPorts := ["X","DY"]
  writePorts := ["DX"]

def kernel (X DY DX : RegionName) (s : Strides) (N B : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := B
  let stride_x_row_value : Int := s.xRow
  let stride_x_col_value : Int := s.xCol
  let stride_dy_row_value : Int := s.dyRow
  let stride_dy_col_value : Int := s.dyCol
  let stride_dx_row_value : Int := s.dxRow
  let stride_dx_col_value : Int := s.dxCol
  let N_value : Int := N
  with_triton_call sourceSignature call_args
    [stride_x_row := stride_x_row_value, stride_x_col := stride_x_col_value,
     stride_dy_row := stride_dy_row_value, stride_dy_col := stride_dy_col_value,
     stride_dx_row := stride_dx_row_value, stride_dx_col := stride_dx_col_value, N := N_value] in
  triton {
    row = tl.program_id(0).to(tl.int64)
    X += row * stride_x_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols * stride_x_col, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols * stride_dy_col, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols * stride_dx_col, dx, mask=mask)
  }

def indexCommutedKernel (X DY DX : RegionName) (s : Strides) (N B : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := B
  let stride_x_row_value : Int := s.xRow
  let stride_x_col_value : Int := s.xCol
  let stride_dy_row_value : Int := s.dyRow
  let stride_dy_col_value : Int := s.dyCol
  let stride_dx_row_value : Int := s.dxRow
  let stride_dx_col_value : Int := s.dxCol
  let N_value : Int := N
  with_triton_call sourceSignature call_args
    [stride_x_row := stride_x_row_value, stride_x_col := stride_x_col_value,
     stride_dy_row := stride_dy_row_value, stride_dy_col := stride_dy_col_value,
     stride_dx_row := stride_dx_row_value, stride_dx_col := stride_dx_col_value, N := N_value] in
  triton {
    row = tl.program_id(0).to(tl.int64)
    X += stride_x_row * row
    DX += stride_dx_row * row
    DY += stride_dy_row * row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols * stride_x_col, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols * stride_dy_col, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols * stride_dx_col, dx, mask=mask)
  }

def source (s : Strides) (N B : Nat) : ComputeKernel :=
  kernel "X" "DY" "DX" s N B

def indexCommuted (s : Strides) (N B : Nat) : ComputeKernel :=
  indexCommutedKernel "X" "DY" "DX" s N B

end VeriTile.Bench.Repairs.L2NormBwd
