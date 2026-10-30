/- Selected source: layer_norm_liger_repaired.py, forward and backward only.
All storage/arithmetic is FP32. Parameters retain actual observed row strides.
The DSL uses natural index carriers; KernelStrides gives the modulo-64 native
pointer recurrence and proves equality at every active access. Empty-program
preludes and after-last increments may wrap; no equality of unused pointers is
claimed. Source floating casts and arithmetic remain explicit below.
-/
import VeriTile.Triton
namespace VeriTile.Bench.Repairs.LayerNormLiger.Forward
open VeriTile.Triton
/-- Complete repaired forward body. Storage declarations retain typed cells;
the denominator conversion materializes Python's implicit fp32 scalar
conversion. Variance padding is explicitly zeroed before the reduction. -/
def source (dtype : FloatDType)
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Y_ptr = DTYPE, X_ptr = DTYPE, W_ptr = DTYPE, B_ptr = DTYPE, Mean_ptr = DTYPE, RSTD_ptr = DTYPE
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y_ptr += row_idx * $(Y_row_stride)
  X_ptr += row_idx * $(X_row_stride)
  Mean_ptr += row_idx * $(Mean_row_stride)
  RSTD_ptr += row_idx * $(RSTD_row_stride)
  X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
  W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
  B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
  mean = tl.sum(X_row, axis=0) / ($((n_cols : ℝ))).to(tl.float32)
  var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / ($((n_cols : ℝ))).to(tl.float32)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean_ptr, (mean).to(DTYPE))
  tl.store(RSTD_ptr, (rstd).to(DTYPE))
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y_ptr + col_offsets, (Y_row).to(DTYPE), mask=mask)
}

/-- Common-subexpression candidate: compute the FP32 centered vector once
and reuse the identical rounded value for variance and affine output. -/
def candidateSource (dtype : FloatDType)
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Y_ptr = DTYPE, X_ptr = DTYPE, W_ptr = DTYPE, B_ptr = DTYPE, Mean_ptr = DTYPE, RSTD_ptr = DTYPE
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y_ptr += row_idx * $(Y_row_stride)
  X_ptr += row_idx * $(X_row_stride)
  Mean_ptr += row_idx * $(Mean_row_stride)
  RSTD_ptr += row_idx * $(RSTD_row_stride)
  X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
  W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
  B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
  mean = tl.sum(X_row, axis=0) / ($((n_cols : ℝ))).to(tl.float32)
  centered = X_row - mean
  var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / ($((n_cols : ℝ))).to(tl.float32)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean_ptr, (mean).to(DTYPE))
  tl.store(RSTD_ptr, (rstd).to(DTYPE))
  Y_row = centered * rstd * W_row + B_row
  tl.store(Y_ptr + col_offsets, (Y_row).to(DTYPE), mask=mask)
}

structure Parameters where
  (y x weight bias mean rstd : RegionName)
  (yStride xStride weightStride biasStride meanStride rstdStride columns block : Nat)
  epsilon : ℝ

def kernel (p : Parameters) : ComputeKernel :=
  source .fp32 p.y p.yStride p.x p.xStride p.weight p.weightStride p.bias p.biasStride
    p.mean p.meanStride p.rstd p.rstdStride p.columns p.epsilon p.block

def candidateKernel (p : Parameters) : ComputeKernel :=
  candidateSource .fp32 p.y p.yStride p.x p.xStride p.weight p.weightStride p.bias p.biasStride
    p.mean p.meanStride p.rstd p.rstdStride p.columns p.epsilon p.block

end VeriTile.Bench.Repairs.LayerNormLiger.Forward

namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton

/-- Whole repaired backward source. Only dtype declarations and explicit
denominator conversions make implicit source typing visible. -/
def source (dtype : FloatDType) (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, Mean = DTYPE, RSTD = DTYPE,
    DX = DTYPE, DW = DTYPE, DB = DTYPE, DY = DTYPE
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(n_rows))
  cols = tl.arange(0, $(BLOCK_SIZE))
  mask = cols < $(n_cols)
  dw_row = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  db_row = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  X += row_start * $(stride_x)
  Mean += row_start
  RSTD += row_start
  DX += row_start * $(stride_dx)
  DY += row_start * $(stride_dy)
  for _row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    dy = tl.load(DY + cols, mask=mask, other=0.0)
    mean = tl.load(Mean)
    rstd = tl.load(RSTD)
    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / ($((n_cols : ℝ))).to(tl.float32)
    c2 = tl.sum(wdy, axis=0) / ($((n_cols : ℝ))).to(tl.float32)
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    tl.store(DX + cols, (dx).to(DTYPE), mask=mask)
    dw_row += dy * x_hat
    db_row += dy
    X += $(stride_x)
    Mean += $(1)
    RSTD += $(1)
    DX += $(stride_dx)
    DY += $(stride_dy)
  }
  tl.store(DW + row_block_id * $(stride_dw) + cols, (dw_row).to(DTYPE), mask=mask)
  tl.store(DB + row_block_id * $(stride_db) + cols, (db_row).to(DTYPE), mask=mask)
}

/-- Loop-invariant conversion candidate: bind the original FP32 denominator
once before the row interval, then reuse it for both divisions in every row. -/
def candidateSource (dtype : FloatDType) (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, Mean = DTYPE, RSTD = DTYPE,
    DX = DTYPE, DW = DTYPE, DB = DTYPE, DY = DTYPE
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(n_rows))
  cols = tl.arange(0, $(BLOCK_SIZE))
  mask = cols < $(n_cols)
  dw_row = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  db_row = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  X += row_start * $(stride_x)
  Mean += row_start
  RSTD += row_start
  DX += row_start * $(stride_dx)
  DY += row_start * $(stride_dy)
  denominator = ($((n_cols : ℝ))).to(tl.float32)
  for _row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    dy = tl.load(DY + cols, mask=mask, other=0.0)
    mean = tl.load(Mean)
    rstd = tl.load(RSTD)
    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / denominator
    c2 = tl.sum(wdy, axis=0) / denominator
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    tl.store(DX + cols, (dx).to(DTYPE), mask=mask)
    dw_row += dy * x_hat
    db_row += dy
    X += $(stride_x)
    Mean += $(1)
    RSTD += $(1)
    DX += $(stride_dx)
    DY += $(stride_dy)
  }
  tl.store(DW + row_block_id * $(stride_dw) + cols, (dw_row).to(DTYPE), mask=mask)
  tl.store(DB + row_block_id * $(stride_db) + cols, (db_row).to(DTYPE), mask=mask)
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
