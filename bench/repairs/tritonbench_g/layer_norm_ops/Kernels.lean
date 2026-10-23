import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LayerNormOps
open VeriTile.Triton

/-! Complete selected source bodies. FP32 storage annotations expose the default
input/output profile. Plain backward uses the actual saved forward Mean; the
original standalone missing-Mean supplied-zero cut is retained outside this
composed contract. No operation, row-loop, residual store or recompute branch is
replaced by a result oracle. -/
def forwardSource (dtype : FloatDType)
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel := triton {
  tl.region X = DTYPE, Y = DTYPE, W = DTYPE, B = DTYPE, RESIDUAL = DTYPE,
    RESIDUAL_OUT = DTYPE, Mean = DTYPE, Rstd = DTYPE
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  if HAS_RESIDUAL {
    RESIDUAL += row * $(stride_res_row)
  }
  if STORE_RESIDUAL_OUT {
    RESIDUAL_OUT += row * $(stride_res_out_row)
  }
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  if HAS_RESIDUAL {
    residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x += residual
  }
  if STORE_RESIDUAL_OUT {
    tl.store(RESIDUAL_OUT + cols, (x).to(DTYPE), mask=cols < $(N))
  }
  tl.if not IS_RMS_NORM {
    mean = tl.sum(x, axis=0) / $(N)
    tl.store(Mean + row, (mean).to(DTYPE))
    xbar = tl.where(cols < $(N), x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  } else {
    xbar = tl.where(cols < $(N), x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  }
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, (rstd).to(DTYPE))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  tl.if HAS_BIAS {
    b = tl.load(B + cols, mask=mask).to(tl.float32)
  }
  x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
  y = (x_hat * w + b if HAS_BIAS else x_hat * w)
  tl.store(Y + cols, (y).to(DTYPE), mask=mask)
}

def backwardSource (dtype : FloatDType)
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row stride_mean stride_rstd M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, B = DTYPE, Y = DTYPE, DY = DTYPE, DX = DTYPE,
    DW = DTYPE, DB = DTYPE, DRESIDUAL = DTYPE, DRESIDUAL_IN = DTYPE, Mean = DTYPE, Rstd = DTYPE
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  X += row_start * $(stride_x_row)
  if HAS_DRESIDUAL {
    DRESIDUAL += row_start * $(stride_dres_row)
  }
  if STORE_DRESIDUAL {
    DRESIDUAL_IN += row_start * $(stride_dres_in_row)
  }
  DY += row_start * $(stride_dy_row)
  DX += row_start * $(stride_dx_row)
  if RECOMPUTE_OUTPUT {
    Y += row_start * $(stride_y_row)
  }
  w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
  if RECOMPUTE_OUTPUT {
    if HAS_BIAS {
      b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    }
  }
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    db = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  }
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(M))
  for row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    tl.if not IS_RMS_NORM {
      mean = tl.load(Mean + row * $(stride_mean))
    }
    rstd = tl.load(Rstd + row * $(stride_rstd))
    xhat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
    xhat = tl.where(mask, xhat, 0.0)
    if RECOMPUTE_OUTPUT {
      y = (xhat * w + b if HAS_BIAS else xhat * w)
      tl.store(Y + cols, (y).to(DTYPE), mask=mask)
    }
    wdy = w * dy
    dw += dy * xhat
    if HAS_BIAS {
      db += dy
    }
    tl.if not IS_RMS_NORM {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
    } else {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      dx = (wdy - xhat * c1) * rstd
    }
    if HAS_DRESIDUAL {
      dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
      dx += dres
    }
    if STORE_DRESIDUAL {
      tl.store(DRESIDUAL_IN + cols, (dx).to(DTYPE), mask=mask)
    }
    tl.store(DX + cols, (dx).to(DTYPE), mask=mask)
    X += $(stride_x_row)
    if HAS_DRESIDUAL {
      DRESIDUAL += $(stride_dres_row)
    }
    if STORE_DRESIDUAL {
      DRESIDUAL_IN += $(stride_dres_in_row)
    }
    if RECOMPUTE_OUTPUT {
      Y += $(stride_y_row)
    }
    DY += $(stride_dy_row)
    DX += $(stride_dx_row)
  }
  tl.store(DW + row_block_id * $(N) + cols, (dw).to(DTYPE), mask=mask)
  if HAS_BIAS {
    tl.store(DB + row_block_id * $(N) + cols, (db).to(DTYPE), mask=mask)
  }
}


/-- Reuse the same column mask throughout the complete forward invocation. -/
def forwardMaskReusedSource (dtype : FloatDType)
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel := triton {
  tl.region X = DTYPE, Y = DTYPE, W = DTYPE, B = DTYPE, RESIDUAL = DTYPE,
    RESIDUAL_OUT = DTYPE, Mean = DTYPE, Rstd = DTYPE
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  if HAS_RESIDUAL {
    RESIDUAL += row * $(stride_res_row)
  }
  if STORE_RESIDUAL_OUT {
    RESIDUAL_OUT += row * $(stride_res_out_row)
  }
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
  if HAS_RESIDUAL {
    residual = tl.load(RESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
    x += residual
  }
  if STORE_RESIDUAL_OUT {
    tl.store(RESIDUAL_OUT + cols, (x).to(DTYPE), mask=mask)
  }
  tl.if not IS_RMS_NORM {
    mean = tl.sum(x, axis=0) / $(N)
    tl.store(Mean + row, (mean).to(DTYPE))
    xbar = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  } else {
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  }
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, (rstd).to(DTYPE))
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  tl.if HAS_BIAS {
    b = tl.load(B + cols, mask=mask).to(tl.float32)
  }
  x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
  y = (x_hat * w + b if HAS_BIAS else x_hat * w)
  tl.store(Y + cols, (y).to(DTYPE), mask=mask)
}

/-- Reuse the computed row start when forming the bounded loop end. -/
def backwardRowStartReusedSource (dtype : FloatDType)
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row stride_mean stride_rstd M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, B = DTYPE, Y = DTYPE, DY = DTYPE, DX = DTYPE,
    DW = DTYPE, DB = DTYPE, DRESIDUAL = DTYPE, DRESIDUAL_IN = DTYPE, Mean = DTYPE, Rstd = DTYPE
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  X += row_start * $(stride_x_row)
  if HAS_DRESIDUAL {
    DRESIDUAL += row_start * $(stride_dres_row)
  }
  if STORE_DRESIDUAL {
    DRESIDUAL_IN += row_start * $(stride_dres_in_row)
  }
  DY += row_start * $(stride_dy_row)
  DX += row_start * $(stride_dx_row)
  if RECOMPUTE_OUTPUT {
    Y += row_start * $(stride_y_row)
  }
  w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
  if RECOMPUTE_OUTPUT {
    if HAS_BIAS {
      b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    }
  }
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    db = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  }
  row_end = min(row_start + $(rows_per_program), $(M))
  for row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    tl.if not IS_RMS_NORM {
      mean = tl.load(Mean + row * $(stride_mean))
    }
    rstd = tl.load(Rstd + row * $(stride_rstd))
    xhat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
    xhat = tl.where(mask, xhat, 0.0)
    if RECOMPUTE_OUTPUT {
      y = (xhat * w + b if HAS_BIAS else xhat * w)
      tl.store(Y + cols, (y).to(DTYPE), mask=mask)
    }
    wdy = w * dy
    dw += dy * xhat
    if HAS_BIAS {
      db += dy
    }
    tl.if not IS_RMS_NORM {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
    } else {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      dx = (wdy - xhat * c1) * rstd
    }
    if HAS_DRESIDUAL {
      dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
      dx += dres
    }
    if STORE_DRESIDUAL {
      tl.store(DRESIDUAL_IN + cols, (dx).to(DTYPE), mask=mask)
    }
    tl.store(DX + cols, (dx).to(DTYPE), mask=mask)
    X += $(stride_x_row)
    if HAS_DRESIDUAL {
      DRESIDUAL += $(stride_dres_row)
    }
    if STORE_DRESIDUAL {
      DRESIDUAL_IN += $(stride_dres_in_row)
    }
    if RECOMPUTE_OUTPUT {
      Y += $(stride_y_row)
    }
    DY += $(stride_dy_row)
    DX += $(stride_dx_row)
  }
  tl.store(DW + row_block_id * $(N) + cols, (dw).to(DTYPE), mask=mask)
  if HAS_BIAS {
    tl.store(DB + row_block_id * $(N) + cols, (db).to(DTYPE), mask=mask)
  }
}


structure ForwardParameters where
  (x y weight bias residual residualOut mean rstd : RegionName)
  (xStride yStride residualStride residualOutStride columns block : Nat)
  epsilon : ℝ
  (rms hasResidual storeResidual hasBias : Bool)

structure BackwardParameters where
  (x weight bias y dy dx dw db dres dresIn mean rstd : RegionName)
  (xStride yStride dyStride dxStride dresStride dresInStride meanStride rstdStride : Nat)
  (rows columns rowsPerProgram block : Nat)
  epsilon : ℝ
  (rms hasDres storeDres hasBias recompute : Bool)

def forwardKernel (p : ForwardParameters) := forwardSource .fp32
  p.x p.y p.weight p.bias p.residual p.residualOut p.mean p.rstd
  p.xStride p.yStride p.residualStride p.residualOutStride p.columns p.block p.epsilon
  p.rms p.hasResidual p.storeResidual p.hasBias

def backwardKernel (p : BackwardParameters) := backwardSource .fp32
  p.x p.weight p.bias p.y p.dy p.dx p.dw p.db p.dres p.dresIn p.mean p.rstd
  p.xStride p.yStride p.dyStride p.dxStride p.dresStride p.dresInStride
  p.meanStride p.rstdStride p.rows p.columns p.rowsPerProgram p.block p.epsilon
  p.rms p.hasDres p.storeDres p.hasBias p.recompute

def forwardMaskReusedKernel (p : ForwardParameters) := forwardMaskReusedSource .fp32
  p.x p.y p.weight p.bias p.residual p.residualOut p.mean p.rstd
  p.xStride p.yStride p.residualStride p.residualOutStride p.columns p.block p.epsilon
  p.rms p.hasResidual p.storeResidual p.hasBias

def backwardRowStartReusedKernel (p : BackwardParameters) := backwardRowStartReusedSource .fp32
  p.x p.weight p.bias p.y p.dy p.dx p.dw p.db p.dres p.dresIn p.mean p.rstd
  p.xStride p.yStride p.dyStride p.dxStride p.dresStride p.dresInStride
  p.meanStride p.rstdStride p.rows p.columns p.rowsPerProgram p.block p.epsilon
  p.rms p.hasDres p.storeDres p.hasBias p.recompute

end VeriTile.Bench.Repairs.LayerNormOps
