import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.swiglu_fwd.Kernels

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("OUT",.fp32)]
  scalars := [("stride_x_row",.integer),("stride_y_row",.integer),("stride_out_row",.integer),("ncols",.integer)]
  constexpr := ["BLOCK_N"]
  readPorts := ["X", "Y"]
  writePorts := ["OUT"]


/-- Historical independently typed storage family, retained for compatibility. -/
abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SwigluFwd.typedKernel

/-- Complete selected original FP32 source, including explicit input casts,
sigmoid, left-associated products, source masks and kernel ports. -/
def kernel (X Y OUT : RegionName) (sx sy so cols block : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := block
  with_triton_call callSignature call_args [stride_x_row := sx, stride_y_row := sy, stride_out_row := so, ncols := cols] in
  triton {
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)
  }

/-- Independent value on the two original last-axis halves. -/
noncomputable def formula (x y : ℝ) : ℝ := x*y/(1+Real.exp (-x))

/-- Complete existing candidate: commute only the three integer row-stride
products, preserving FP32 loads, casts, sigmoid and multiplication order. -/
def indexCommutedKernel (X Y OUT : RegionName) (sx sy so cols block : Nat) : ComputeKernel :=
  let BLOCK_N : Nat := block
  with_triton_call callSignature call_args [stride_x_row := sx, stride_y_row := sy, stride_out_row := so, ncols := cols] in
  triton {
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += stride_x_row * row
    Y += stride_y_row * row
    OUT += stride_out_row * row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)
  }

end VeriTile.Bench.Repairs.SwigluFwd
