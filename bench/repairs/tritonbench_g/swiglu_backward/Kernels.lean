import bench.tritonbench_g.swiglu_backward.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Raw selected FP32 source and the existing integer-address candidate.
Both recomputation branches remain symbolic; call metadata stays outside source.
The candidate Python counterpart is explicitly attributed in candidate_kernels.py. -/
namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SwigluBackward.typedKernel

def callSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("DOUT",.fp32),("OUT",.fp32),("DX",.fp32),("DY",.fp32)]
  scalars := [("stride_x_row",.i32),("stride_y_row",.i32),("stride_dout_row",.i32),("stride_out_row",.i32),("stride_dx_row",.i32),("stride_dy_row",.i32),("ncols",.i32)]
  constexpr := ["BLOCK_N","RECOMPUTE_OUTPUT"]
  readPorts := ["X","Y","DOUT"]
  writePorts := ["DX","DY","OUT"]

def kernel (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy cols block : Nat) (recompute : Bool) : ComputeKernel :=
  let BLOCK_N := block
  let RECOMPUTE_OUTPUT := recompute
  with_triton_call callSignature call_args [stride_x_row := sx, stride_y_row := sy, stride_dout_row := sd, stride_out_row := so, stride_dx_row := sdx, stride_dy_row := sdy, ncols := cols] in
  triton {
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    DOUT += row * stride_dout_row
    if RECOMPUTE_OUTPUT:
        OUT += row * stride_out_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < ncols, other=0.).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < ncols)
    tl.store(DY + cols, dy, mask=cols < ncols)
    if RECOMPUTE_OUTPUT:
        out = x * x_sigmoid * y
        tl.store(OUT + cols, out, mask=cols < ncols)
  }

def commutedKernel (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy cols block : Nat) (recompute : Bool) : ComputeKernel :=
  let BLOCK_N := block
  let RECOMPUTE_OUTPUT := recompute
  with_triton_call callSignature call_args [stride_x_row := sx, stride_y_row := sy, stride_dout_row := sd, stride_out_row := so, stride_dx_row := sdx, stride_dy_row := sdy, ncols := cols] in
  triton {
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += stride_x_row * row
    Y += stride_y_row * row
    DOUT += stride_dout_row * row
    if RECOMPUTE_OUTPUT:
        OUT += stride_out_row * row
    DX += stride_dx_row * row
    DY += stride_dy_row * row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < ncols, other=0.).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < ncols)
    tl.store(DY + cols, dy, mask=cols < ncols)
    if RECOMPUTE_OUTPUT:
        out = x * x_sigmoid * y
        tl.store(OUT + cols, out, mask=cols < ncols)
  }

end VeriTile.Bench.Repairs.SwigluBackward
