import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LayerNormOps
open VeriTile.Triton

/-! Complete original Python bodies and the existing two candidate rewrites.
The existing caller validates FP32 storage. Scalar invocation values and
constexpr flags are supplied outside
the quotations. The composed plain backward call supplies the saved Mean.
RMS mode supplies Mean=None through the caller presence binding. -/

def forwardSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("W",.fp32),("B",.fp32),("RESIDUAL",.fp32),("RESIDUAL_OUT",.fp32),("Mean",.fp32),("Rstd",.fp32)]
  scalars := [("stride_x_row",.i32),("stride_y_row",.i32),("stride_res_row",.i32),("stride_res_out_row",.i32),("N",.i32),("eps",.fp32)]
  typedFP32Storage := Bool.true
  constexpr := ["BLOCK_N","IS_RMS_NORM","HAS_RESIDUAL","STORE_RESIDUAL_OUT","HAS_BIAS"]
  readPorts := ["X","W","B","RESIDUAL"]
  writePorts := ["Y","RESIDUAL_OUT","Mean","Rstd"]

def backwardSignature : CallSignature where
  pointers := [("X",.fp32),("W",.fp32),("B",.fp32),("Y",.fp32),("DY",.fp32),("DX",.fp32),("DW",.fp32),("DB",.fp32),("DRESIDUAL",.fp32),("DRESIDUAL_IN",.fp32),("Mean",.fp32),("Rstd",.fp32)]
  scalars := [("stride_x_row",.i32),("stride_y_row",.i32),("stride_dy_row",.i32),("stride_dx_row",.i32),("stride_dres_row",.i32),("stride_dres_in_row",.i32),("stride_mean",.i32),("stride_rstd",.i32),("M",.i32),("N",.i32),("rows_per_program",.i32),("eps",.fp32)]
  typedFP32Storage := Bool.true
  constexpr := ["BLOCK_N","IS_RMS_NORM","HAS_DRESIDUAL","STORE_DRESIDUAL","HAS_BIAS","RECOMPUTE_OUTPUT"]
  optionalPointers := ["Mean"]
  readPorts := ["X","W","B","DY","DRESIDUAL","Mean","Rstd"]
  writePorts := ["Y","DX","DW","DB","DRESIDUAL_IN"]

def forwardSource
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel :=
  with_triton_call forwardSignature call_args [stride_x_row := stride_x_row,stride_y_row := stride_y_row,stride_res_row := stride_res_row,stride_res_out_row := stride_res_out_row,N := N,eps := eps] in
  triton {
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }

def backwardSource
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row stride_mean stride_rstd M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel :=
  let meanPresent : Bool := !IS_RMS_NORM
  with_triton_call backwardSignature call_args [stride_x_row := stride_x_row,stride_y_row := stride_y_row,stride_dy_row := stride_dy_row,stride_dx_row := stride_dx_row,stride_dres_row := stride_dres_row,stride_dres_in_row := stride_dres_in_row,stride_mean := stride_mean,stride_rstd := stride_rstd,M := M,N := N,rows_per_program := rows_per_program,eps := eps] pointer_presence [Mean := meanPresent] in
  triton {
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row * stride_mean) if Mean is not None else 0.0  # Use zero when Mean is absent.
        rstd = tl.load(Rstd + row * stride_rstd)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
  }

def forwardMaskReusedSource
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel :=
  with_triton_call forwardSignature call_args [stride_x_row := stride_x_row,stride_y_row := stride_y_row,stride_res_row := stride_res_row,stride_res_out_row := stride_res_out_row,N := N,eps := eps] in
  triton {
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=mask)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }

def backwardRowStartReusedSource
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row stride_mean stride_rstd M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel :=
  let meanPresent : Bool := !IS_RMS_NORM
  with_triton_call backwardSignature call_args [stride_x_row := stride_x_row,stride_y_row := stride_y_row,stride_dy_row := stride_dy_row,stride_dx_row := stride_dx_row,stride_dres_row := stride_dres_row,stride_dres_in_row := stride_dres_in_row,stride_mean := stride_mean,stride_rstd := stride_rstd,M := M,N := N,rows_per_program := rows_per_program,eps := eps] pointer_presence [Mean := meanPresent] in
  triton {
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row * stride_mean) if Mean is not None else 0.0  # Use zero when Mean is absent.
        rstd = tl.load(Rstd + row * stride_rstd)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
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

def forwardKernel (p : ForwardParameters) := forwardSource
  p.x p.y p.weight p.bias p.residual p.residualOut p.mean p.rstd
  p.xStride p.yStride p.residualStride p.residualOutStride p.columns p.block p.epsilon
  p.rms p.hasResidual p.storeResidual p.hasBias

def backwardKernel (p : BackwardParameters) := backwardSource
  p.x p.weight p.bias p.y p.dy p.dx p.dw p.db p.dres p.dresIn p.mean p.rstd
  p.xStride p.yStride p.dyStride p.dxStride p.dresStride p.dresInStride
  p.meanStride p.rstdStride p.rows p.columns p.rowsPerProgram p.block p.epsilon
  p.rms p.hasDres p.storeDres p.hasBias p.recompute

def forwardMaskReusedKernel (p : ForwardParameters) := forwardMaskReusedSource
  p.x p.y p.weight p.bias p.residual p.residualOut p.mean p.rstd
  p.xStride p.yStride p.residualStride p.residualOutStride p.columns p.block p.epsilon
  p.rms p.hasResidual p.storeResidual p.hasBias

def backwardRowStartReusedKernel (p : BackwardParameters) := backwardRowStartReusedSource
  p.x p.weight p.bias p.y p.dy p.dx p.dw p.db p.dres p.dresIn p.mean p.rstd
  p.xStride p.yStride p.dyStride p.dxStride p.dresStride p.dresInStride
  p.meanStride p.rstdStride p.rows p.columns p.rowsPerProgram p.block p.epsilon
  p.rms p.hasDres p.storeDres p.hasBias p.recompute

end VeriTile.Bench.Repairs.LayerNormOps
