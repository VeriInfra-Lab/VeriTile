import VeriTile.Triton.DSL.FP32CallValues

/-! Complete pasted repaired Python and the existing integer-product candidate.
All invocation typing and the checked arange size remain outside the body. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdSelectedSource
open VeriTile.Triton
noncomputable section
def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("W",.fp32),("B",.fp32),("RESIDUAL",.fp32),("X1",.fp32),("W1",.fp32),("B1",.fp32),("Y1",.fp32),("RESIDUAL_OUT",.fp32),("ROWSCALE",.fp32),("SEEDS",.i64),("DROPOUT_MASK",.boolean),("Mean",.fp32),("Rstd",.fp32)]
  scalars := [("eps",.fp32),("dropout_p",.fp32),("stride_x_row",.integer),("stride_y_row",.integer),("stride_res_row",.integer),("stride_res_out_row",.integer),("stride_x1_row",.integer),("stride_y1_row",.integer),("M",.integer),("N",.integer)]
  constexpr := ["IS_RMS_NORM","BLOCK_N","HAS_RESIDUAL","STORE_RESIDUAL_OUT","HAS_BIAS","HAS_DROPOUT","STORE_DROPOUT_MASK","HAS_ROWSCALE","HAS_X1","HAS_W1","HAS_B1"]
  readPorts := ["X","W","B","RESIDUAL","X1","W1","B1","ROWSCALE","SEEDS"]
  writePorts := ["Y","Y1","RESIDUAL_OUT","DROPOUT_MASK","Mean","Rstd"]

def originalKernel (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
          (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
          (stride_x_row stride_y_row stride_res_row stride_res_out_row
            stride_x1_row stride_y1_row M N : Nat)
          (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
          (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
          (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
            STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) : ComputeKernel :=
          with_triton_call signature call_args [eps := eps, dropout_p := dropout_p, stride_x_row := stride_x_row, stride_y_row := stride_y_row, stride_res_row := stride_res_row, stride_res_out_row := stride_res_out_row, stride_x1_row := stride_x1_row, stride_y1_row := stride_y1_row, M := M, N := N] integer_args [] arange_args [BLOCK_N := hBlock] in triton {
    row = tl.program_id(0).to(tl.int64)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
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
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)
  }

def indexCommutedKernel (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
          (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
          (stride_x_row stride_y_row stride_res_row stride_res_out_row
            stride_x1_row stride_y1_row M N : Nat)
          (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
          (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
          (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
            STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) : ComputeKernel :=
          with_triton_call signature call_args [eps := eps, dropout_p := dropout_p, stride_x_row := stride_x_row, stride_y_row := stride_y_row, stride_res_row := stride_res_row, stride_res_out_row := stride_res_out_row, stride_x1_row := stride_x1_row, stride_y1_row := stride_y1_row, M := M, N := N] integer_args [] arange_args [BLOCK_N := hBlock] in triton {
    row = tl.program_id(0).to(tl.int64)
    X += stride_x_row * row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
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
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)
  }

end
end VeriTile.Bench.Repairs.LayerNormFwdSelectedSource
