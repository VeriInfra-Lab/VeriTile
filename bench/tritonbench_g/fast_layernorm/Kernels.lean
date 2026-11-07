import VeriTile.Triton
import VeriTile.Triton.DSL.ScalarInt64While

namespace VeriTile.Bench.Repairs.FastLayernorm
open VeriTile.Triton
noncomputable section

def defaultPrecisionBits : Nat := 32
def affineBlockSize : Nat := 128
def maximumFusedWidth : Nat := 65536

def forwardCallSignature : CallSignature where
  pointers := [("Y",.fp32), ("X",.fp32), ("W",.fp32), ("b",.fp32), ("r",.fp32), ("mu",.fp32)]
  scalars := [("Y_row_stride",.integer), ("X_row_stride",.integer), ("n_cols",.integer), ("eps",.fp32)]
  constexpr := ["BLOCK_SIZE", "HAS_BIAS"]
  readPorts := ["X", "W", "b"]
  writePorts := ["Y", "r", "mu"]

/-- Complete original repaired Python body; all caller metadata is outside the quotation. -/
def forwardBodyPastedKernel (Y X W b r mu : RegionName) (B : Nat) (hasBias : Bool) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  let HAS_BIAS := hasBias
  with_triton_signature forwardCallSignature pointer_offsets [Y := 0, X := 0, W := 0, b := 0, r := 0, mu := 0] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
    Y += row_idx * tl.cast(Y_row_stride, tl.int64)
    X += row_idx * tl.cast(X_row_stride, tl.int64)
    r += row_idx
    mu += row_idx
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
    else:
        b_row = tl.full((BLOCK_SIZE,), 0, tl.float32)
    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = tl.where(mask, X_row - mean_X, 0.0)
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)
  }

/-- Complete existing equivalence candidate; all caller metadata is outside the quotation. -/
def forwardCandidateKernel (Y X W b r mu : RegionName) (B : Nat) (hasBias : Bool) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  let HAS_BIAS := hasBias
  with_triton_signature forwardCallSignature pointer_offsets [Y := 0, X := 0, W := 0, b := 0, r := 0, mu := 0] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
    Y += row_idx * tl.cast(Y_row_stride, tl.int64)
    X += tl.cast(X_row_stride, tl.int64) * row_idx
    r += row_idx
    mu += row_idx
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
    else:
        b_row = tl.full((BLOCK_SIZE,), 0, tl.float32)
    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = tl.where(mask, X_row - mean_X, 0.0)
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)
  }

def backwardCallSignature : CallSignature where
  pointers := [("dX",.fp32), ("dY",.fp32), ("X",.fp32), ("W",.fp32), ("r",.fp32), ("mu",.fp32)]
  scalars := [("dX_row_stride",.integer), ("dY_row_stride",.integer), ("X_row_stride",.integer), ("n_cols",.integer), ("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["dY", "X", "W", "r", "mu"]
  writePorts := ["dX"]

/-- Complete original repaired Python body; all caller metadata is outside the quotation. -/
def backwardBodyPastedKernel (dX dY X W r mu : RegionName) (B : Nat) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  with_triton_signature backwardCallSignature pointer_offsets [dX := 0, dY := 0, X := 0, W := 0, r := 0, mu := 0] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
    dX += row_idx * tl.cast(dX_row_stride, tl.int64)
    dY += row_idx * tl.cast(dY_row_stride, tl.int64)
    X += row_idx * tl.cast(X_row_stride, tl.int64)
    r += row_idx
    mu += row_idx
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    inv_var = tl.load(r).to(tl.float32)
    mean = tl.load(mu).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    # Selecting the product prevents padded 0*NaN/Inf from entering the sum.
    product = tl.where(mask, dY_W * normed, 0.0)
    dX_row = dY_W - tl.sum(dY_W, axis=0) / n_cols - normed * tl.sum(product, axis=0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dX + col_offsets, dX_row, mask=mask)
  }

/-- Complete existing equivalence candidate; all caller metadata is outside the quotation. -/
def backwardCandidateKernel (dX dY X W r mu : RegionName) (B : Nat) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  with_triton_signature backwardCallSignature pointer_offsets [dX := 0, dY := 0, X := 0, W := 0, r := 0, mu := 0] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
    dX += row_idx * tl.cast(dX_row_stride, tl.int64)
    dY += row_idx * tl.cast(dY_row_stride, tl.int64)
    X += tl.cast(X_row_stride, tl.int64) * row_idx
    r += row_idx
    mu += row_idx
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    inv_var = tl.load(r).to(tl.float32)
    mean = tl.load(mu).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    # Selecting the product prevents padded 0*NaN/Inf from entering the sum.
    product = tl.where(mask, dY_W * normed, 0.0)
    dX_row = dY_W - tl.sum(dY_W, axis=0) / n_cols - normed * tl.sum(product, axis=0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dX + col_offsets, dX_row, mask=mask)
  }

def affineCallSignature : CallSignature where
  pointers := [("dW",.fp32), ("db",.fp32), ("dY",.fp32), ("X",.fp32), ("r",.fp32), ("mu",.fp32)]
  scalars := [("n_rows",.integer), ("n_cols",.integer), ("weight_count",.integer), ("bias_count",.integer)]
  constexpr := ["BLOCK_SIZE", "HAS_BIAS"]
  readPorts := ["dY", "X", "r", "mu"]
  writePorts := ["dW", "db"]

/-- Complete original repaired Python body; all caller metadata is outside the quotation. -/
def affineBodyPastedKernel (dW db dY X r mu : RegionName) (hasBias : Bool) : ComputeKernel :=
  let BLOCK_SIZE : Nat := 128
  let HAS_BIAS := hasBias
  with_triton_signature affineCallSignature pointer_offsets [dW := 0, db := 0, dY := 0, X := 0, r := 0, mu := 0] in triton {
    n_rows = tl.cast(n_rows, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    weight_count = tl.cast(weight_count, tl.int64)
    bias_count = tl.cast(bias_count, tl.int64)
    columns = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    active = columns < n_cols
    dw = tl.full((BLOCK_SIZE,), 0, tl.float32)
    dbias = tl.full((BLOCK_SIZE,), 0, tl.float32)
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
        offsets = row * tl.cast(n_cols, tl.int64) + tl.where(active, columns, 0)
        dy = tl.load(dY + offsets, mask=active, other=0).to(tl.float32)
        x = tl.load(X + offsets, mask=active, other=0).to(tl.float32)
        inv = tl.load(r + row).to(tl.float32)
        avg = tl.load(mu + row).to(tl.float32)
        z = (x - avg) * inv
        # All requested gradient cells are written, including unused affine tails.
        dw += tl.where(active, dy * z, 0.0)
        dbias += dy
        row += tl.full((), 1, tl.int64)
    tl.store(dW + columns, dw, mask=columns < weight_count)
    if HAS_BIAS:
        tl.store(db + columns, dbias, mask=columns < bias_count)
  }

/-- Complete existing equivalence candidate; all caller metadata is outside the quotation. -/
def affineCandidateKernel (dW db dY X r mu : RegionName) (hasBias : Bool) : ComputeKernel :=
  let BLOCK_SIZE : Nat := 128
  let HAS_BIAS := hasBias
  with_triton_signature affineCallSignature pointer_offsets [dW := 0, db := 0, dY := 0, X := 0, r := 0, mu := 0] in triton {
    n_rows = tl.cast(n_rows, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    weight_count = tl.cast(weight_count, tl.int64)
    bias_count = tl.cast(bias_count, tl.int64)
    columns = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    active = columns < n_cols
    dw = tl.full((BLOCK_SIZE,), 0, tl.float32)
    dbias = tl.full((BLOCK_SIZE,), 0, tl.float32)
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
        offsets = row * tl.cast(n_cols, tl.int64) + columns
        dy = tl.load(dY + offsets, mask=active, other=0).to(tl.float32)
        x = tl.load(X + offsets, mask=active, other=0).to(tl.float32)
        inv = tl.load(r + row).to(tl.float32)
        avg = tl.load(mu + row).to(tl.float32)
        z = (x - avg) * inv
        # All requested gradient cells are written, including unused affine tails.
        dw += tl.where(active, dy * z, 0.0)
        dbias += dy
        row += tl.full((), 1, tl.int64)
    tl.store(dW + columns, dw, mask=columns < weight_count)
    if HAS_BIAS:
        tl.store(db + columns, dbias, mask=columns < bias_count)
  }

def forwardKernel (Y X W b r mu : RegionName) (N B : Nat) (eps : ℝ) (hasBias : Bool) : ComputeKernel :=
  (forwardBodyPastedKernel Y X W b r mu B hasBias).withScalarArguments [
    ScalarArgument.integer "Y_row_stride" (N : Int),
    ScalarArgument.integer "X_row_stride" (N : Int),
    ScalarArgument.integer "n_cols" (N : Int),
    ScalarArgument.fp32 "eps" eps]

def forwardIndexCommuted (Y X W b r mu : RegionName) (N B : Nat) (eps : ℝ) (hasBias : Bool) : ComputeKernel :=
  (forwardCandidateKernel Y X W b r mu B hasBias).withScalarArguments [
    ScalarArgument.integer "Y_row_stride" (N : Int),
    ScalarArgument.integer "X_row_stride" (N : Int),
    ScalarArgument.integer "n_cols" (N : Int),
    ScalarArgument.fp32 "eps" eps]

def backwardKernel (dX dY X W r mu : RegionName) (N B : Nat) : ComputeKernel :=
  (backwardBodyPastedKernel dX dY X W r mu B).withScalarArguments [
    ScalarArgument.integer "dX_row_stride" (N : Int),
    ScalarArgument.integer "dY_row_stride" (N : Int),
    ScalarArgument.integer "X_row_stride" (N : Int),
    ScalarArgument.integer "n_cols" (N : Int)]

def backwardIndexCommuted (dX dY X W r mu : RegionName) (N B : Nat) : ComputeKernel :=
  (backwardCandidateKernel dX dY X W r mu B).withScalarArguments [
    ScalarArgument.integer "dX_row_stride" (N : Int),
    ScalarArgument.integer "dY_row_stride" (N : Int),
    ScalarArgument.integer "X_row_stride" (N : Int),
    ScalarArgument.integer "n_cols" (N : Int)]

def affineKernel (dW db dY X r mu : RegionName) (rows N weightCount biasCount : Nat) (hasBias : Bool) : ComputeKernel :=
  (affineBodyPastedKernel dW db dY X r mu hasBias).withScalarArguments [
    ScalarArgument.integer "n_rows" (rows : Int),
    ScalarArgument.integer "n_cols" (N : Int),
    ScalarArgument.integer "weight_count" (weightCount : Int),
    ScalarArgument.integer "bias_count" (biasCount : Int)]

def affineDirectOffsets (dW db dY X r mu : RegionName) (rows N weightCount biasCount : Nat) (hasBias : Bool) : ComputeKernel :=
  (affineCandidateKernel dW db dY X r mu hasBias).withScalarArguments [
    ScalarArgument.integer "n_rows" (rows : Int),
    ScalarArgument.integer "n_cols" (N : Int),
    ScalarArgument.integer "weight_count" (weightCount : Int),
    ScalarArgument.integer "bias_count" (biasCount : Int)]

end
end VeriTile.Bench.Repairs.FastLayernorm
