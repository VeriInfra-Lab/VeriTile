/- Selected source: layer_norm_liger_repaired.py, forward and backward only.
Both quotations paste the original function bodies. The selected Python host
requires FP32 storage; invocation metadata supplies this dtype and scalar
arguments outside the quotations. All source casts, pointer updates, masks,
reductions and loop statements remain visible. Candidate bodies are authored
counterparts of the existing centered-vector reuse and denominator-hoist
rewrites, recorded separately in layer_norm_liger_candidates.py.
-/
import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
namespace VeriTile.Bench.Repairs.LayerNormLiger.Forward
open VeriTile.Triton

def forwardSignature : CallSignature where
  typedFP32Storage := Bool.true
  pointers := [("Y_ptr",.fp32),("X_ptr",.fp32),("W_ptr",.fp32),("B_ptr",.fp32),("Mean_ptr",.fp32),("RSTD_ptr",.fp32)]
  scalars := [("Y_row_stride",.integer),("X_row_stride",.integer),("W_row_stride",.integer),("B_row_stride",.integer),("Mean_row_stride",.integer),("RSTD_row_stride",.integer),("n_cols",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X_ptr", "W_ptr", "B_ptr"]
  writePorts := ["Y_ptr", "Mean_ptr", "RSTD_ptr"]

/-- Complete repaired forward Python body, including explicit source casts
and masked variance padding. FP32 caller metadata stays outside the quote. -/
def forwardSource
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
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

def source := forwardSource

/-- Common-subexpression candidate: compute the FP32 centered vector once
and reuse the identical rounded value for variance and affine output. -/
def forwardCandidateSource
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
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

def candidateSource := forwardCandidateSource

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

namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton

def backwardSignature : CallSignature where
  typedFP32Storage := Bool.true
  pointers := [("X_ptr",.fp32),("W_ptr",.fp32),("Mean_ptr",.fp32),("RSTD_ptr",.fp32),("DX_ptr",.fp32),("DW_ptr",.fp32),("DB_ptr",.fp32),("DY_ptr",.fp32)]
  scalars := [("stride_x",.integer),("stride_dx",.integer),("stride_dw",.integer),("stride_db",.integer),("stride_dy",.integer),("n_rows",.integer),("n_cols",.integer)]
  constexpr := ["rows_per_program", "BLOCK_SIZE", "dtype"]
  readPorts := ["X_ptr", "W_ptr", "Mean_ptr", "RSTD_ptr", "DY_ptr"]
  writePorts := ["DX_ptr", "DW_ptr", "DB_ptr"]


/-- Complete repaired backward Python body. The caller supplies the source
FP32 dtype; the signed scalar loop step remains in the quoted body. -/
def backwardSource (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) : ComputeKernel := 
  let dtype : FloatDType := .fp32
  let X_ptr : RegionName := X
  let W_ptr : RegionName := W
  let Mean_ptr : RegionName := Mean
  let RSTD_ptr : RegionName := RSTD
  let DX_ptr : RegionName := DX
  let DW_ptr : RegionName := DW
  let DB_ptr : RegionName := DB
  let DY_ptr : RegionName := DY
  with_triton_call backwardSignature call_args [stride_x := stride_x,stride_dx := stride_dx,stride_dw := stride_dw,stride_db := stride_db,stride_dy := stride_dy,n_rows := n_rows,n_cols := n_cols] in triton {
    stride_x = tl.cast(stride_x, tl.int64)
    stride_dx = tl.cast(stride_dx, tl.int64)
    stride_dw = tl.cast(stride_dw, tl.int64)
    stride_db = tl.cast(stride_db, tl.int64)
    stride_dy = tl.cast(stride_dy, tl.int64)
    n_rows = tl.cast(n_rows, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_block_id = tl.program_id(0).to(tl.int64)
    group = tl.full((), rows_per_program, tl.int64)
    one = tl.full((), 1, tl.int64)
    row_start = row_block_id * group
    row_end = min((row_block_id + one) * group, n_rows)
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    for _ in range(row_start, row_end, one):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)
  }

def source := backwardSource

/-- Loop-invariant conversion candidate: bind the original FP32 denominator
once before the row interval, then reuse it for both divisions in every row. -/
def backwardCandidateSource (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) : ComputeKernel := 
  let dtype : FloatDType := .fp32
  let X_ptr : RegionName := X
  let W_ptr : RegionName := W
  let Mean_ptr : RegionName := Mean
  let RSTD_ptr : RegionName := RSTD
  let DX_ptr : RegionName := DX
  let DW_ptr : RegionName := DW
  let DB_ptr : RegionName := DB
  let DY_ptr : RegionName := DY
  with_triton_call backwardSignature call_args [stride_x := stride_x,stride_dx := stride_dx,stride_dw := stride_dw,stride_db := stride_db,stride_dy := stride_dy,n_rows := n_rows,n_cols := n_cols] in triton {
    stride_x = tl.cast(stride_x, tl.int64)
    stride_dx = tl.cast(stride_dx, tl.int64)
    stride_dw = tl.cast(stride_dw, tl.int64)
    stride_db = tl.cast(stride_db, tl.int64)
    stride_dy = tl.cast(stride_dy, tl.int64)
    n_rows = tl.cast(n_rows, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_block_id = tl.program_id(0).to(tl.int64)
    group = tl.full((), rows_per_program, tl.int64)
    one = tl.full((), 1, tl.int64)
    row_start = row_block_id * group
    row_end = min((row_block_id + one) * group, n_rows)
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    denominator = n_cols.to(tl.float32)
    for _ in range(row_start, row_end, one):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / denominator
        c2 = tl.sum(wdy, axis=0) / denominator
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)
  }

def candidateSource := backwardCandidateSource

structure Parameters where
  (x weight mean rstd dx dw db dy : RegionName)
  (xStride dxStride dwStride dbStride dyStride rows columns rowsPerProgram block : Nat)

def kernel (p : Parameters) : ComputeKernel := source p.x p.weight p.mean p.rstd p.dx p.dw p.db p.dy
  p.xStride p.dxStride p.dwStride p.dbStride p.dyStride p.rows p.columns p.rowsPerProgram p.block


def candidateKernel (p : Parameters) : ComputeKernel :=
  candidateSource p.x p.weight p.mean p.rstd p.dx p.dw p.db p.dy
    p.xStride p.dxStride p.dwStride p.dbStride p.dyStride p.rows p.columns p.rowsPerProgram p.block

end VeriTile.Bench.Repairs.LayerNormLiger.Backward
