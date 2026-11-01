import VeriTile.Triton.DSL.PythonShapeTuples

namespace VeriTile.Bench.Repairs.LayerNormLiger.Forward
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("Y_ptr",.fp32), ("X_ptr",.fp32), ("W_ptr",.fp32), ("B_ptr",.fp32), ("Mean_ptr",.fp32), ("RSTD_ptr",.fp32)]
  scalars := [("Y_row_stride",.i64), ("X_row_stride",.i64), ("W_row_stride",.i64), ("B_row_stride",.i64), ("Mean_row_stride",.i64), ("RSTD_row_stride",.i64), ("n_cols",.i64), ("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X_ptr", "W_ptr", "B_ptr"]
  writePorts := ["Y_ptr", "Mean_ptr", "RSTD_ptr"]

/-- Verbatim repaired Python forward body. Pointer dtypes and runtime scalar
bindings are caller metadata outside the quotation. Both divisions retain the
original implicit integer-to-FP32 conversion; no source cast is added. -/
def source
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel :=
  let Y_row_stride_argument : Int := Y_row_stride
  let X_row_stride_argument : Int := X_row_stride
  let W_row_stride_argument : Int := W_row_stride
  let B_row_stride_argument : Int := B_row_stride
  let Mean_row_stride_argument : Int := Mean_row_stride
  let RSTD_row_stride_argument : Int := RSTD_row_stride
  let n_cols_argument : Int := n_cols
  with_triton_call callSignature call_args [Y_row_stride := Y_row_stride_argument, X_row_stride := X_row_stride_argument, W_row_stride := W_row_stride_argument, B_row_stride := B_row_stride_argument, Mean_row_stride := Mean_row_stride_argument, RSTD_row_stride := RSTD_row_stride_argument, n_cols := n_cols_argument, eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = (X_row - mean) * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
}

/-- Common-subexpression candidate: compute the FP32 centered vector once
and reuse the identical rounded value for variance and affine output. -/
def candidateSource
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel :=
  let Y_row_stride_argument : Int := Y_row_stride
  let X_row_stride_argument : Int := X_row_stride
  let W_row_stride_argument : Int := W_row_stride
  let B_row_stride_argument : Int := B_row_stride
  let Mean_row_stride_argument : Int := Mean_row_stride
  let RSTD_row_stride_argument : Int := RSTD_row_stride
  let n_cols_argument : Int := n_cols
  with_triton_call callSignature call_args [Y_row_stride := Y_row_stride_argument, X_row_stride := X_row_stride_argument, W_row_stride := W_row_stride_argument, B_row_stride := B_row_stride_argument, Mean_row_stride := Mean_row_stride_argument, RSTD_row_stride := RSTD_row_stride_argument, n_cols := n_cols_argument, eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    centered = X_row - mean
    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = centered * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
}

structure Parameters where
  (y x weight bias mean rstd : RegionName)
  (yStride xStride weightStride biasStride meanStride rstdStride columns block : Nat)
  epsilon : ℝ

def kernel (p : Parameters) : ComputeKernel :=
  source p.y p.yStride p.x p.xStride p.weight p.weightStride p.bias p.biasStride
    p.mean p.meanStride p.rstd p.rstdStride p.columns p.epsilon p.block

def candidateKernel (p : Parameters) : ComputeKernel :=
  candidateSource p.y p.yStride p.x p.xStride p.weight p.weightStride p.bias p.biasStride
    p.mean p.meanStride p.rstd p.rstdStride p.columns p.epsilon p.block

end VeriTile.Bench.Repairs.LayerNormLiger.Forward
