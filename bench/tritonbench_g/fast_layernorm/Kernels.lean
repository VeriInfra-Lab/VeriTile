import VeriTile.Triton

namespace VeriTile.Bench.Repairs.FastLayernorm
open VeriTile.Triton

def defaultPrecisionBits : Nat := 32
def affineBlockSize : Nat := 128
def maximumFusedWidth : Nat := 65536

/-- Actual FP32 forward source. Integer widening is erased here and attached
to the finite native addresses in Proofs/Addresses. -/
def forwardKernel (Y X W b r mu : RegionName) (N B : Nat) (eps : ℝ) (hasBias : Bool) : ComputeKernel := triton {
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32, b = tl.float32, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(B))
  mask = col_offsets < $(N)
  Y += row_idx * $(N)
  X += row_idx * $(N)
  r += row_idx
  mu += row_idx
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  if $((hasBias : Bool)) {
    b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
  } else {
    b_row = tl.zeros([$(B)], dtype=tl.float32)
  }
  mean_X = tl.sum(X_row, axis=0) / $(N)
  XX = tl.where(mask, X_row - mean_X, 0.0)
  row_var = tl.sum(XX * XX, axis=0) / $(N)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var.to(tl.float32))
  tl.store(mu, mean_X.to(tl.float32))
  output = (XX * inv_var) * W_row + b_row
  tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)
}

def backwardKernel (dX dY X W r mu : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region dX = tl.float32, dY = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(B))
  mask = col_offsets < $(N)
  dX += row_idx * $(N)
  dY += row_idx * $(N)
  X += row_idx * $(N)
  r += row_idx
  mu += row_idx
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  mean = tl.load(mu).to(tl.float32)
  normed = (X_row - mean) * inv_var
  dY_W = dY_row * W_row
  product = tl.where(mask, dY_W * normed, 0.0)
  dX_row = dY_W - tl.sum(dY_W, axis=0) / $(N) - normed * tl.sum(product, axis=0) / $(N)
  dX_row = dX_row * inv_var
  tl.store(dX + col_offsets, dX_row.to(tl.float32), mask=mask)
}

/-- The increasing row loop normalizes the source's int64 while(0..rows).
Each iteration has the same load/update body and unit successor. -/
def affineKernel (dW db dY X r mu : RegionName) (rows N weightCount biasCount : Nat)
    (hasBias : Bool) : ComputeKernel := triton {
  tl.region dW = tl.float32, db = tl.float32, dY = tl.float32, X = tl.float32, r = tl.float32, mu = tl.float32
  columns = tl.program_id(0) * $(128) + tl.arange(0, $(128))
  active = columns < $(N)
  dw = tl.zeros([128], dtype=tl.float32)
  dbias = tl.zeros([128], dtype=tl.float32)
  for row in range($(rows)) {
    offsets = row * $(N) + tl.where(active, columns, $(0))
    dy = tl.load(dY + offsets, mask=active, other=0).to(tl.float32)
    x = tl.load(X + offsets, mask=active, other=0).to(tl.float32)
    inv = tl.load(r + row).to(tl.float32)
    avg = tl.load(mu + row).to(tl.float32)
    z = (x - avg) * inv
    dw += tl.where(active, dy * z, 0.0)
    dbias += dy
  }
  tl.store(dW + columns, dw.to(tl.float32), mask=columns < $(weightCount))
  if $((hasBias : Bool)) {
    tl.store(db + columns, dbias.to(tl.float32), mask=columns < $(biasCount))
  }
}

/-- Preserved X row-index commutation applied to the current repaired source. -/
def forwardIndexCommuted (Y X W b r mu : RegionName) (N B : Nat) (eps : ℝ) (hasBias : Bool) : ComputeKernel := triton {
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32, b = tl.float32, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(B))
  mask = col_offsets < $(N)
  Y += row_idx * $(N)
  X += $(N) * row_idx
  r += row_idx
  mu += row_idx
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  if $((hasBias : Bool)) {
    b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
  } else {
    b_row = tl.zeros([$(B)], dtype=tl.float32)
  }
  mean_X = tl.sum(X_row, axis=0) / $(N)
  XX = tl.where(mask, X_row - mean_X, 0.0)
  row_var = tl.sum(XX * XX, axis=0) / $(N)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var.to(tl.float32))
  tl.store(mu, mean_X.to(tl.float32))
  output = (XX * inv_var) * W_row + b_row
  tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)
}

def backwardIndexCommuted (dX dY X W r mu : RegionName) (N B : Nat) : ComputeKernel := triton {
  tl.region dX = tl.float32, dY = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(B))
  mask = col_offsets < $(N)
  dX += row_idx * $(N)
  dY += row_idx * $(N)
  X += $(N) * row_idx
  r += row_idx
  mu += row_idx
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  mean = tl.load(mu).to(tl.float32)
  normed = (X_row - mean) * inv_var
  dY_W = dY_row * W_row
  product = tl.where(mask, dY_W * normed, 0.0)
  dX_row = dY_W - tl.sum(dY_W, axis=0) / $(N) - normed * tl.sum(product, axis=0) / $(N)
  dX_row = dX_row * inv_var
  tl.store(dX + col_offsets, dX_row.to(tl.float32), mask=mask)
}


/-- Remove the redundant per-row column select: inactive addresses are masked
at both loads. All FP operations, row order and output masks are retained. -/
def affineDirectOffsets (dW db dY X r mu : RegionName) (rows N weightCount biasCount : Nat)
    (hasBias : Bool) : ComputeKernel := triton {
  tl.region dW = tl.float32, db = tl.float32, dY = tl.float32, X = tl.float32, r = tl.float32, mu = tl.float32
  columns = tl.program_id(0) * $(128) + tl.arange(0, $(128))
  active = columns < $(N)
  dw = tl.zeros([128], dtype=tl.float32)
  dbias = tl.zeros([128], dtype=tl.float32)
  for row in range($(rows)) {
    offsets = row * $(N) + columns
    dy = tl.load(dY + offsets, mask=active, other=0).to(tl.float32)
    x = tl.load(X + offsets, mask=active, other=0).to(tl.float32)
    inv = tl.load(r + row).to(tl.float32)
    avg = tl.load(mu + row).to(tl.float32)
    z = (x - avg) * inv
    dw += tl.where(active, dy * z, 0.0)
    dbias += dy
  }
  tl.store(dW + columns, dw.to(tl.float32), mask=columns < $(weightCount))
  if $((hasBias : Bool)) {
    tl.store(db + columns, dbias.to(tl.float32), mask=columns < $(biasCount))
  }
}

end VeriTile.Bench.Repairs.FastLayernorm
