"""Explicit Python sources for the existing FastLayernorm equivalence candidates.
Forward/backward commute only the X row-address product. Affine removes the
inactive column select under the existing two load masks. The repaired originals
are unchanged in fast_layernorm.py.
"""
import triton
import triton.language as tl

@triton.jit
def layernorm_forward_index_commuted(Y, Y_row_stride, X, X_row_stride, W, b, r, mu,
                      n_cols, eps, BLOCK_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr):
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

@triton.jit
def layernorm_backward_index_commuted(dX, dX_row_stride, dY, dY_row_stride, X, X_row_stride,
                       W, r, mu, n_cols, eps, BLOCK_SIZE: tl.constexpr):
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

@triton.jit
def layernorm_affine_backward_direct_offsets(dW, db, dY, X, r, mu, n_rows, n_cols,
                              weight_count, bias_count, BLOCK_SIZE: tl.constexpr,
                              HAS_BIAS: tl.constexpr):
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

