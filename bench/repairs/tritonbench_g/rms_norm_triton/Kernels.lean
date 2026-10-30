import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Exact original Python body and the retained row-product candidate.
The caller supplies the contiguous strides and pointer/scalar types outside
these quotations. Runtime epsilon has its own ordinary register interface;
the literal verification interface retains the same complete Python body. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("Y",.fp32),("X",.fp32),("W",.fp32)]
  scalars := [("y_stride_r",.i32),("y_stride_c",.i32),("x_stride_r",.i32),("x_stride_c",.i32),("N",.i32),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W"]
  writePorts := ["Y"]

def kernel (Y X W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  let BLOCK_SIZE := B
  let unitStride := (1 : Nat)
  with_triton_call callSignature call_args [y_stride_r := N, y_stride_c := unitStride, x_stride_r := N, x_stride_c := unitStride, N := N, eps := eps] in
  triton {
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
  }

def runtimeKernel (Y X W : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_SIZE := B
  let unitStride := (1 : Nat)
  with_triton_call callSignature call_args [y_stride_r := N, y_stride_c := unitStride, x_stride_r := N, x_stride_c := unitStride, N := N] in
  triton {
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
  }

def indexCommutedKernel (Y X W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  let BLOCK_SIZE := B
  let unitStride := (1 : Nat)
  with_triton_call callSignature call_args [y_stride_r := N, y_stride_c := unitStride, x_stride_r := N, x_stride_c := unitStride, N := N, eps := eps] in
  triton {
    pid = tl.program_id(0)
    Y += y_stride_r * pid
    X += x_stride_r * pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
  }

def runtimeCandidate (Y X W : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_SIZE := B
  let unitStride := (1 : Nat)
  with_triton_call callSignature call_args [y_stride_r := N, y_stride_c := unitStride, x_stride_r := N, x_stride_c := unitStride, N := N] in
  triton {
    pid = tl.program_id(0)
    Y += y_stride_r * pid
    X += x_stride_r * pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
  }

end VeriTile.Bench.Repairs.RmsNormTriton
