import bench.repairs.tritonbench_g.layer_norm_fwd.Kernels

/-! The positive-tile repair retains the complete ordinary JIT sources, including all optional streams, random
operations, statistics, masks and destination conversion sites. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Source
open VeriTile.Triton
abbrev Flags := VeriTile.Bench.Repairs.LayerNormFwd.Source.Flags
abbrev Regions := VeriTile.Bench.Repairs.LayerNormFwd.Source.Regions
abbrev Strides := VeriTile.Bench.Repairs.LayerNormFwd.Source.Strides
abbrev sourceType := VeriTile.Bench.Repairs.LayerNormFwd.Source.sourceType

/-- Complete existing FP32 source, retaining every optional stream and store. -/
def kernel : sourceType :=
  fun f r stride M N eps p block hBlock =>
    (fun (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
          (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
          (stride_x_row stride_y_row stride_res_row stride_res_out_row
            stride_x1_row stride_y1_row M N : Nat)
          (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
          (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
          (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
            STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) =>
          triton {
      tl.region X = tl.float32, Y = tl.float32, W = tl.float32, B = tl.float32
      tl.region RESIDUAL = tl.float32, X1 = tl.float32, W1 = tl.float32, B1 = tl.float32
      tl.region Y1 = tl.float32, RESIDUAL_OUT = tl.float32, ROWSCALE = tl.float32
      tl.region SEEDS = tl.int64, DROPOUT_MASK = tl.int1, Mean = tl.float32, Rstd = tl.float32
      row = tl.program_id(0)
      X += row * $(stride_x_row)
      Y += row * $(stride_y_row)
      if HAS_RESIDUAL {
        RESIDUAL += row * $(stride_res_row)
      }
      if STORE_RESIDUAL_OUT {
        RESIDUAL_OUT += row * $(stride_res_out_row)
      }
      if HAS_X1 {
        X1 += row * $(stride_x1_row)
      }
      if HAS_W1 {
        Y1 += row * $(stride_y1_row)
      }
      BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hBlock)
      cols = tl.arange(0, BLOCK_N_arg)
      x = tl.load(X + cols, mask=cols < $((N : Nat)), other=0.0).to(tl.float32)
      if HAS_ROWSCALE {
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
      }
      if HAS_DROPOUT {
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > $(dropout_p)
        x = tl.where(keep_mask, x / (1.0 - $(dropout_p)), 0.0)
        if STORE_DROPOUT_MASK {
          tl.store(DROPOUT_MASK + row * $((N : Nat)) + cols, keep_mask, mask=cols < $((N : Nat)))
        }
      }
      if HAS_X1 {
        x1 = tl.load(X1 + cols, mask=cols < $((N : Nat)), other=0.0).to(tl.float32)
        if HAS_ROWSCALE {
          rowscale = tl.load(ROWSCALE + $((M : Nat)) + row).to(tl.float32)
          x1 *= rowscale
        }
        if HAS_DROPOUT {
          keep_mask = tl.rand(tl.load(SEEDS + $((M : Nat)) + row).to(tl.uint32), cols, n_rounds=7) > $(dropout_p)
          x1 = tl.where(keep_mask, x1 / (1.0 - $(dropout_p)), 0.0)
          if STORE_DROPOUT_MASK {
            tl.store(DROPOUT_MASK + ($((M : Nat)) + row) * $((N : Nat)) + cols, keep_mask, mask=cols < $((N : Nat)))
          }
        }
        x += x1
      }
      if HAS_RESIDUAL {
        residual = tl.load(RESIDUAL + cols, mask=cols < $((N : Nat)), other=0.0).to(tl.float32)
        x += residual
      }
      if STORE_RESIDUAL_OUT {
        tl.store(RESIDUAL_OUT + cols, (x).to(tl.float32), mask=cols < $((N : Nat)))
      }
      tl.if not IS_RMS_NORM {
        mean = tl.sum(x, axis=0) / $((N : Nat))
        tl.store(Mean + row, (mean).to(tl.float32))
        xbar = tl.where(cols < $((N : Nat)), x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / $((N : Nat))
      } else {
        xbar = tl.where(cols < $((N : Nat)), x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / $((N : Nat))
      }
      rstd = 1 / tl.sqrt(var + $(eps))
      tl.store(Rstd + row, (rstd).to(tl.float32))
      mask = cols < $((N : Nat))
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      tl.if HAS_BIAS {
        b = tl.load(B + cols, mask=mask).to(tl.float32)
      }
      x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
      y = (x_hat * w + b if HAS_BIAS else x_hat * w)
      tl.store(Y + cols, (y).to(tl.float32), mask=mask)
      if HAS_W1 {
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        tl.if HAS_B1 {
          b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        }
        y1 = (x_hat * w1 + b1 if HAS_B1 else x_hat * w1)
        tl.store(Y1 + cols, (y1).to(tl.float32), mask=mask)
      }
    })
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1

/-- Existing candidate: only the initial integer X row/stride product is commuted. -/
def indexCommuted : sourceType :=
  fun f r stride M N eps p block hBlock =>
    (fun (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
          (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
          (stride_x_row stride_y_row stride_res_row stride_res_out_row
            stride_x1_row stride_y1_row M N : Nat)
          (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
          (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
          (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
            STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) =>
          triton {
      tl.region X = tl.float32, Y = tl.float32, W = tl.float32, B = tl.float32
      tl.region RESIDUAL = tl.float32, X1 = tl.float32, W1 = tl.float32, B1 = tl.float32
      tl.region Y1 = tl.float32, RESIDUAL_OUT = tl.float32, ROWSCALE = tl.float32
      tl.region SEEDS = tl.int64, DROPOUT_MASK = tl.int1, Mean = tl.float32, Rstd = tl.float32
      row = tl.program_id(0)
      X += $((stride_x_row : Nat)) * row
      Y += row * $(stride_y_row)
      if HAS_RESIDUAL {
        RESIDUAL += row * $(stride_res_row)
      }
      if STORE_RESIDUAL_OUT {
        RESIDUAL_OUT += row * $(stride_res_out_row)
      }
      if HAS_X1 {
        X1 += row * $(stride_x1_row)
      }
      if HAS_W1 {
        Y1 += row * $(stride_y1_row)
      }
      BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hBlock)
      cols = tl.arange(0, BLOCK_N_arg)
      x = tl.load(X + cols, mask=cols < $((N : Nat)), other=0.0).to(tl.float32)
      if HAS_ROWSCALE {
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
      }
      if HAS_DROPOUT {
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > $(dropout_p)
        x = tl.where(keep_mask, x / (1.0 - $(dropout_p)), 0.0)
        if STORE_DROPOUT_MASK {
          tl.store(DROPOUT_MASK + row * $((N : Nat)) + cols, keep_mask, mask=cols < $((N : Nat)))
        }
      }
      if HAS_X1 {
        x1 = tl.load(X1 + cols, mask=cols < $((N : Nat)), other=0.0).to(tl.float32)
        if HAS_ROWSCALE {
          rowscale = tl.load(ROWSCALE + $((M : Nat)) + row).to(tl.float32)
          x1 *= rowscale
        }
        if HAS_DROPOUT {
          keep_mask = tl.rand(tl.load(SEEDS + $((M : Nat)) + row).to(tl.uint32), cols, n_rounds=7) > $(dropout_p)
          x1 = tl.where(keep_mask, x1 / (1.0 - $(dropout_p)), 0.0)
          if STORE_DROPOUT_MASK {
            tl.store(DROPOUT_MASK + ($((M : Nat)) + row) * $((N : Nat)) + cols, keep_mask, mask=cols < $((N : Nat)))
          }
        }
        x += x1
      }
      if HAS_RESIDUAL {
        residual = tl.load(RESIDUAL + cols, mask=cols < $((N : Nat)), other=0.0).to(tl.float32)
        x += residual
      }
      if STORE_RESIDUAL_OUT {
        tl.store(RESIDUAL_OUT + cols, (x).to(tl.float32), mask=cols < $((N : Nat)))
      }
      tl.if not IS_RMS_NORM {
        mean = tl.sum(x, axis=0) / $((N : Nat))
        tl.store(Mean + row, (mean).to(tl.float32))
        xbar = tl.where(cols < $((N : Nat)), x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / $((N : Nat))
      } else {
        xbar = tl.where(cols < $((N : Nat)), x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / $((N : Nat))
      }
      rstd = 1 / tl.sqrt(var + $(eps))
      tl.store(Rstd + row, (rstd).to(tl.float32))
      mask = cols < $((N : Nat))
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      tl.if HAS_BIAS {
        b = tl.load(B + cols, mask=mask).to(tl.float32)
      }
      x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
      y = (x_hat * w + b if HAS_BIAS else x_hat * w)
      tl.store(Y + cols, (y).to(tl.float32), mask=mask)
      if HAS_W1 {
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        tl.if HAS_B1 {
          b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        }
        y1 = (x_hat * w1 + b1 if HAS_B1 else x_hat * w1)
        tl.store(Y1 + cols, (y1).to(tl.float32), mask=mask)
      }
    })
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Source
