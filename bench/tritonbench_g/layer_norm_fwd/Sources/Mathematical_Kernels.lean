import VeriTile.Triton.DSL

/-!
Source-audit target: _layer_norm_fwd_1pass_kernel
Source-audit Lean: layerNormFwdSource%
Source-audit args: input=tl.float32, output=tl.float32, weight=tl.float32, bias=tl.float32, residual=tl.float32, input1=tl.float32, weight1=tl.float32, bias1=tl.float32, residualOut=tl.float32, rowscale=tl.float32, version=original
Source-audit materializations: {"kind":"guarded_constexpr_identity_v1","source_sha256":"27dfccf23671d5b9c6f4b749cef39ddf69079793c486a833cbae9c57fe3a4de4","lean":{"source":"layerNormFwdSource%","args":{"input":"tl.float32","output":"tl.float32","weight":"tl.float32","bias":"tl.float32","residual":"tl.float32","input1":"tl.float32","weight1":"tl.float32","bias1":"tl.float32","residualOut":"tl.float32","rowscale":"tl.float32","version":"original"}},"regions":{"X":"tl.float32","Y":"tl.float32","W":"tl.float32","B":"tl.float32","RESIDUAL":"tl.float32","X1":"tl.float32","W1":"tl.float32","B1":"tl.float32","Y1":"tl.float32","RESIDUAL_OUT":"tl.float32","ROWSCALE":"tl.float32","Mean":"tl.float32","Rstd":"tl.float32","SEEDS":"tl.int64","DROPOUT_MASK":"tl.int1"},"binding":{"python":"BLOCK_N","local":"BLOCK_N_arg","kind":"constexpr_identity","adapter":"_root_.VeriTile.Triton.Random.i32ArangeSize","adapter_sha256":"61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04","proof":"hBlock","before":"cols"}}
This profile compares the complete selected fp32 source AST, including all ten
symbolic flags. It is not acceptance of every storage/native profile or proof replay.
-/


/-!
Full source transcription of `_layer_norm_fwd_1pass_kernel`.

This macro intentionally retains the source int64 seed memory followed by the
uint32 cast. The frontend implements the cast by reduction modulo 2^32 while
retaining the int64 storage type. `Proofs.SourceChecks` checks fp32 lowering
without projection failure; other storage profiles remain separate obligations.

The candidate changes only the order of the integer row/stride factors in X's
initial pointer increment. Floating evaluation and store order remain exact.
The floating storage dtypes are macro arguments; fp32 is not used as a proxy for
all host-admitted storage combinations. The source's native store conversions
are explicit at each destination.
-/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

open Lean VeriTile.Triton

syntax "layerNormFwdSource%" "[" tritonDType "," tritonDType ","
  tritonDType "," tritonDType "," tritonDType "," tritonDType ","
  tritonDType "," tritonDType "," tritonDType "," tritonDType "," ident "]" : term

set_option hygiene false in
macro_rules
  | `(layerNormFwdSource%[$input:tritonDType, $output:tritonDType,
      $weight:tritonDType, $bias:tritonDType, $residual:tritonDType,
      $input1:tritonDType, $weight1:tritonDType, $bias1:tritonDType,
      $residualOut:tritonDType, $rowscale:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let xAdvance : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| X += $((stride_x_row : Nat)) * row)
      else `(tritonStmt| X += row * $(stride_x_row))
    `(fun (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
      (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
      (stride_x_row stride_y_row stride_res_row stride_res_out_row
        stride_x1_row stride_y1_row M N : Nat)
      (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
      (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
      (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
        STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) =>
      triton {
  tl.region X = $input, Y = $output, W = $weight, B = $bias
  tl.region RESIDUAL = $residual, X1 = $input1, W1 = $weight1, B1 = $bias1
  tl.region Y1 = $output, RESIDUAL_OUT = $residualOut, ROWSCALE = $rowscale
  tl.region SEEDS = tl.int64, DROPOUT_MASK = tl.int1, Mean = tl.float32, Rstd = tl.float32
  row = tl.program_id(0)
  $xAdvance
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
    tl.store(RESIDUAL_OUT + cols, (x).to($residualOut:tritonDType), mask=cols < $((N : Nat)))
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
  tl.store(Y + cols, (y).to($output:tritonDType), mask=mask)
  if HAS_W1 {
    w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
    tl.if HAS_B1 {
      b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
    }
    y1 = (x_hat * w1 + b1 if HAS_B1 else x_hat * w1)
    tl.store(Y1 + cols, (y1).to($output:tritonDType), mask=mask)
  }
})

end VeriTile.Bench.TritonBenchG.LayerNormFwd
