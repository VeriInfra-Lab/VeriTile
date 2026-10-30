import bench.tritonbench_g.layer_norm_liger.Kernels
import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardKernels
namespace VeriTile.Bench.Repairs.LayerNormLiger.Shape
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
@[simp] theorem erase_real_cast (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType :=
  Op.eraseDType_castFloat .real .real e
/-- Native production transcription, with the two explicit padding repairs. -/
def forwardSurface
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
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
  mean = tl.sum(X_row, axis=0) / $(n_cols)
  var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / $(n_cols)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean_ptr, mean)
  tl.store(RSTD_ptr, rstd)
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
}

def backwardSurface
    (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
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
    c1 = tl.sum(x_hat * wdy, axis=0) / $(n_cols)
    c2 = tl.sum(wdy, axis=0) / $(n_cols)
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    tl.store(DX + cols, (dx).to(DX.dtype.element_ty), mask=mask)
    dw_row += dy * x_hat
    db_row += dy
    X += $(stride_x)
    Mean += $(1)
    RSTD += $(1)
    DX += $(stride_dx)
    DY += $(stride_dy)
  }
  tl.store(DW + row_block_id * $(stride_dw) + cols,
    (dw_row).to(DW.dtype.element_ty), mask=mask)
  tl.store(DB + row_block_id * $(stride_db) + cols,
    (db_row).to(DB.dtype.element_ty), mask=mask)
}

theorem forward_shape (p : Forward.Parameters) :
    (Forward.kernel p).eraseDType.surfaceBody =
      (forwardSurface p.y p.yStride p.x p.xStride p.weight p.weightStride
        p.bias p.biasStride p.mean p.meanStride p.rstd p.rstdStride p.columns p.epsilon p.block).eraseDType.surfaceBody := by
  simp [Forward.kernel, Forward.source, forwardSurface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
theorem backward_shape (p : Backward.Parameters) :
    (Backward.kernel p).eraseDType.surfaceBody =
      (backwardSurface p.x p.weight p.mean p.rstd p.dx p.dw p.db p.dy
        p.xStride p.dxStride p.dwStride p.dbStride p.dyStride p.rows p.columns p.rowsPerProgram p.block).eraseDType.surfaceBody := by
  simp [Backward.kernel, Backward.source, backwardSurface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]

#axiomsClean forward_shape
#axiomsClean backward_shape
end VeriTile.Bench.Repairs.LayerNormLiger.Shape
