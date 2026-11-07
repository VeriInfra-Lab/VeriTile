import VeriTile.Triton.DSL.PointerDTypeZeros
namespace VeriTile.Bench.Repairs.LayerNormTriton
open VeriTile.Triton
noncomputable section
def forwardSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("W",.fp32),("B",.fp32),("Mean",.fp32),("Rstd",.fp32)]
  scalars := [("stride",.integer),("N",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W","B"]
  writePorts := ["Y","Mean","Rstd"]
def forwardKernel (X Y W B Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel :=
  with_triton_call forwardSignature call_args [stride := stride, N := N, eps := eps] integer_args [] arange_args [] in triton {
    row = tl.program_id(0).to(tl.int64)
    stride = tl.cast(stride, tl.int64)
    N = tl.cast(N, tl.int64)
    Y += row * stride
    X += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
  }

def groupedBackwardSignature : CallSignature where
  pointers := [("DX",.fp32),("DY",.fp32),("DW",.fp32),("DB",.fp32),("X",.fp32),("W",.fp32),("Mean",.fp32),("Rstd",.fp32)]
  scalars := [("stride",.integer),("M",.integer),("N",.integer)]
  constexpr := ["GROUP_SIZE_M","BLOCK_SIZE_N"]
  readPorts := ["DY","X","W","Mean","Rstd"]
  writePorts := ["DX","DW","DB"]
def groupedBackwardKernel (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call groupedBackwardSignature call_args [stride := stride, M := M, N := N] integer_args [] arange_args [] in triton {
    group = tl.program_id(0).to(tl.int64)
    stride = tl.cast(stride, tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    cols = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    mask = cols < N
    DW = DW + group * N + cols
    DB = DB + group * N + cols
    dw_acc = tl.zeros([BLOCK_SIZE_N], dtype=DW.dtype.element_ty)
    db_acc = tl.zeros([BLOCK_SIZE_N], dtype=DB.dtype.element_ty)
    row = group
    while row < M:
        x = tl.load(X + row * stride + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + row * stride + cols, dx, mask=mask)
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = dy.to(w.dtype)
        if row != group:
            partial_dw += dw_acc
            partial_db += db_acc
        # Retain each original partial-storage rounding in the carried value.
        dw_acc = partial_dw.to(DW.dtype.element_ty)
        db_acc = partial_db.to(DB.dtype.element_ty)
        row += GROUP_SIZE_M
    tl.store(DW, dw_acc, mask=mask)
    tl.store(DB, db_acc, mask=mask)
  }

def finalBackwardSignature : CallSignature where
  pointers := [("DW",.fp32),("DB",.fp32),("FINAL_DW",.fp32),("FINAL_DB",.fp32)]
  scalars := [("M",.integer),("N",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N"]
  readPorts := ["DW","DB"]
  writePorts := ["FINAL_DW","FINAL_DB"]
def finalBackwardKernel (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call finalBackwardSignature call_args [M := M, N := N] integer_args [] arange_args [] in triton {
    pid = tl.program_id(0).to(tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
  }

def forwardCandidateKernel (X Y W B Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel :=
  with_triton_call forwardSignature call_args [stride := stride, N := N, eps := eps] integer_args [] arange_args [] in triton {
    row = tl.program_id(0).to(tl.int64)
    stride = tl.cast(stride, tl.int64)
    N = tl.cast(N, tl.int64)
    Y += row * stride
    X += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    off = 0
    cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    off = 0
    cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x - mean, 0.0)
    _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    off = 0
    cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    tl.store(Y + cols, y, mask=mask)
  }

def groupedBackwardCandidateKernel (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call groupedBackwardSignature call_args [stride := stride, M := M, N := N] integer_args [] arange_args [] in triton {
    group = tl.program_id(0).to(tl.int64)
    stride = tl.cast(stride, tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    cols = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    mask = cols < N
    DW = DW + group * N + cols
    DB = DB + group * N + cols
    dw_acc = tl.zeros([BLOCK_SIZE_N], dtype=DW.dtype.element_ty)
    db_acc = tl.zeros([BLOCK_SIZE_N], dtype=DB.dtype.element_ty)
    row = group
    while row < M:
        x = tl.load(X + row * stride + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + row * stride + cols, dx, mask=mask)
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = dy.to(w.dtype)
        if row != group:
            partial_dw += dw_acc
            partial_db += db_acc
        dw_acc = partial_dw.to(DW.dtype.element_ty)
        db_acc = partial_db.to(DB.dtype.element_ty)
        row += GROUP_SIZE_M
    tl.store(DW, dw_acc, mask=mask)
    tl.store(DB, db_acc, mask=mask)
  }

def finalBackwardCandidateKernel (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call finalBackwardSignature call_args [M := M, N := N] integer_args [] arange_args [] in triton {
    pid = tl.program_id(0).to(tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
  }

end
end VeriTile.Bench.Repairs.LayerNormTriton
