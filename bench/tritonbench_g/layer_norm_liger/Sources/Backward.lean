import VeriTile.Triton.DSL.FloatDTypeConstants

namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("X_ptr",.fp32), ("W_ptr",.fp32), ("Mean_ptr",.fp32), ("RSTD_ptr",.fp32), ("DX_ptr",.fp32), ("DW_ptr",.fp32), ("DB_ptr",.fp32), ("DY_ptr",.fp32)]
  scalars := [("stride_x",.i64), ("stride_dx",.i64), ("stride_dw",.i64), ("stride_db",.i64), ("stride_dy",.i64), ("n_rows",.i64), ("n_cols",.i64)]
  constexpr := ["rows_per_program", "BLOCK_SIZE", "dtype"]
  readPorts := ["X_ptr", "W_ptr", "Mean_ptr", "RSTD_ptr", "DY_ptr"]
  writePorts := ["DX_ptr", "DW_ptr", "DB_ptr"]


/-- Verbatim repaired Python backward body, including signed scalar/lane
casts, scalar int64 constructors, the signed row loop and every output cast.
The selected Python host accepts FP32 input storage. The original output dtype
constexpr remains caller metadata and its casts remain in the source. -/
def source (dtype : FloatDType) (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) : ComputeKernel :=
  let X_ptr := X
  let W_ptr := W
  let Mean_ptr := Mean
  let RSTD_ptr := RSTD
  let DX_ptr := DX
  let DW_ptr := DW
  let DB_ptr := DB
  let DY_ptr := DY
  let stride_x_argument : Int := stride_x
  let stride_dx_argument : Int := stride_dx
  let stride_dw_argument : Int := stride_dw
  let stride_db_argument : Int := stride_db
  let stride_dy_argument : Int := stride_dy
  let n_rows_argument : Int := n_rows
  let n_cols_argument : Int := n_cols
  with_triton_call callSignature call_args [stride_x := stride_x_argument, stride_dx := stride_dx_argument, stride_dw := stride_dw_argument, stride_db := stride_db_argument, stride_dy := stride_dy_argument, n_rows := n_rows_argument, n_cols := n_cols_argument] in triton {
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

/-- Loop-invariant conversion candidate: bind the original FP32 denominator
once before the row interval, then reuse it for both divisions in every row. -/
def candidateSource (dtype : FloatDType) (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) : ComputeKernel :=
  let X_ptr := X
  let W_ptr := W
  let Mean_ptr := Mean
  let RSTD_ptr := RSTD
  let DX_ptr := DX
  let DW_ptr := DW
  let DB_ptr := DB
  let DY_ptr := DY
  let stride_x_argument : Int := stride_x
  let stride_dx_argument : Int := stride_dx
  let stride_dw_argument : Int := stride_dw
  let stride_db_argument : Int := stride_db
  let stride_dy_argument : Int := stride_dy
  let n_rows_argument : Int := n_rows
  let n_cols_argument : Int := n_cols
  with_triton_call callSignature call_args [stride_x := stride_x_argument, stride_dx := stride_dx_argument, stride_dw := stride_dw_argument, stride_db := stride_db_argument, stride_dy := stride_dy_argument, n_rows := n_rows_argument, n_cols := n_cols_argument] in triton {
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

structure Parameters where
  (x weight mean rstd dx dw db dy : RegionName)
  (xStride dxStride dwStride dbStride dyStride rows columns rowsPerProgram block : Nat)

def kernel (p : Parameters) : ComputeKernel := source .fp32 p.x p.weight p.mean p.rstd p.dx p.dw p.db p.dy
  p.xStride p.dxStride p.dwStride p.dbStride p.dyStride p.rows p.columns p.rowsPerProgram p.block


def candidateKernel (p : Parameters) : ComputeKernel :=
  candidateSource .fp32 p.x p.weight p.mean p.rstd p.dx p.dw p.db p.dy
    p.xStride p.dxStride p.dwStride p.dbStride p.dyStride p.rows p.columns p.rowsPerProgram p.block

end VeriTile.Bench.Repairs.LayerNormLiger.Backward
