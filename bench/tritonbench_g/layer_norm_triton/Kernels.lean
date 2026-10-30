import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import bench.tritonbench_g.layer_norm_triton.Proofs.PythonSource

/-! Selected source: `layer_norm_triton_repaired.py`, unchanged SHA256 in
`Source.candidateSHA256`. All data, statistics, partial carries and stores are
FP32. Epsilon is the already ABI-converted scalar, not a replacement literal.
The bounded grouped-row loop spells the source `while row < M; row += G` as
`range(group,M,G)`; Geometry proves its exact rows and termination for G>0.
The DSL address projection uses natural indices; signed64 source bounds are
proved separately in Proofs/Machine. This projection is not a proof of source
or compiler refinement. The complete Python AST remains in Proofs/PythonSource.
-/
namespace VeriTile.Bench.Repairs.LayerNormTriton.TypedSource
open VeriTile.Triton

def forward (X Y W Bias Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[X,W,Bias], outputs=[Y,Mean,Rstd]
    tl.region X = tl.float32, Y = tl.float32, W = tl.float32, Bias = tl.float32,
      Mean = tl.float32, Rstd = tl.float32
    row = tl.program_id(0)
    X += row * $(stride)
    Y += row * $(stride)
    mean_acc = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
    for off in range(0, $(N), $(BLOCK_SIZE)) {
      cols = off + tl.arange(0, $(BLOCK_SIZE))
      a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
      mean_acc += a
    }
    mean = tl.sum(mean_acc, axis=0) / $(N)
    var_acc = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
    for off in range(0, $(N), $(BLOCK_SIZE)) {
      cols = off + tl.arange(0, $(BLOCK_SIZE))
      x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
      x = tl.where(cols < $(N), x - mean, 0.0)
      var_acc += x * x
    }
    variance = tl.sum(var_acc, axis=0) / $(N)
    rstd = 1.0 / tl.sqrt(variance + $(eps))
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, $(N), $(BLOCK_SIZE)) {
      cols = off + tl.arange(0, $(BLOCK_SIZE))
      mask = cols < $(N)
      w = tl.load(W + cols, mask=mask)
      b = tl.load(Bias + cols, mask=mask)
      x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
      x_hat = (x - mean) * rstd
      y = x_hat * w + b
      tl.store(Y + cols, y.to(tl.float32), mask=mask)
    }
  }

def groupedBackward (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[DY,X,W,Mean,Rstd], outputs=[DX,DW,DB]
    tl.region DX = tl.float32, DY = tl.float32, DW = tl.float32, DB = tl.float32,
      X = tl.float32, W = tl.float32, Mean = tl.float32, Rstd = tl.float32
    group = tl.program_id(0)
    cols = tl.arange(0, $(BLOCK_SIZE_N))
    mask = cols < $(N)
    DW += group * $(N) + cols
    DB += group * $(N) + cols
    dw_acc = tl.zeros([$(BLOCK_SIZE_N)], dtype=tl.float32)
    db_acc = tl.zeros([$(BLOCK_SIZE_N)], dtype=tl.float32)
    for row in range(group, $(M), $(GROUP_SIZE_M)) {
      x = tl.load(X + row * $(stride) + cols, mask=mask, other=0.0).to(tl.float32)
      dy = tl.load(DY + row * $(stride) + cols, mask=mask, other=0.0).to(tl.float32)
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      mean = tl.load(Mean + row)
      rstd = tl.load(Rstd + row)
      xhat = (x - mean) * rstd
      wdy = w * dy
      xhat = tl.where(mask, xhat, 0.0)
      wdy = tl.where(mask, wdy, 0.0)
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
      tl.store(DX + row * $(stride) + cols, dx.to(tl.float32), mask=mask)
      partial_dw = (dy * xhat).to(tl.float32)
      partial_db = dy.to(tl.float32)
      if row != group {
        partial_dw += dw_acc
        partial_db += db_acc
      }
      dw_acc = partial_dw.to(tl.float32)
      db_acc = partial_db.to(tl.float32)
    }
    tl.store(DW, dw_acc, mask=mask)
    tl.store(DB, db_acc, mask=mask)
  }

def finalBackward (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  triton {
    tl.region DW = tl.float32, DB = tl.float32, FINAL_DW = tl.float32, FINAL_DB = tl.float32
    pid = tl.program_id(0)
    cols = pid * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    dw = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    db = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    for i in range(0, $(M), $(BLOCK_SIZE_M)) {
      rows = i + tl.arange(0, $(BLOCK_SIZE_M))
      mask = (rows[:, None] < $(M)) & (cols[None, :] < $(N))
      offs = rows[:, None] * $(N) + cols[None, :]
      dw += tl.load(DW + offs, mask=mask, other=0.0)
      db += tl.load(DB + offs, mask=mask, other=0.0)
    }
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(tl.float32), mask=cols < $(N))
    tl.store(FINAL_DB + cols, sum_db.to(tl.float32), mask=cols < $(N))
  }

/-- Existing single-iteration loop expansion, now a complete readable source. -/
def forwardCandidate (X Y W Bias Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[X,W,Bias], outputs=[Y,Mean,Rstd]
    tl.region X = tl.float32, Y = tl.float32, W = tl.float32, Bias = tl.float32,
      Mean = tl.float32, Rstd = tl.float32
    row = tl.program_id(0)
    X += row * $(stride)
    Y += row * $(stride)
    mean_acc = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
    off = $(0)
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    mean_acc += a
    mean = tl.sum(mean_acc, axis=0) / $(N)
    var_acc = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
    off = $(0)
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    var_acc += x * x
    variance = tl.sum(var_acc, axis=0) / $(N)
    rstd = 1.0 / tl.sqrt(variance + $(eps))
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    off = $(0)
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    b = tl.load(Bias + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    tl.store(Y + cols, y.to(tl.float32), mask=mask)
  }


/-- Delay weight loading until immediately before use. -/
def groupedBackwardCandidate (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[DY,X,W,Mean,Rstd], outputs=[DX,DW,DB]
    tl.region DX = tl.float32, DY = tl.float32, DW = tl.float32, DB = tl.float32,
      X = tl.float32, W = tl.float32, Mean = tl.float32, Rstd = tl.float32
    group = tl.program_id(0)
    cols = tl.arange(0, $(BLOCK_SIZE_N))
    mask = cols < $(N)
    DW += group * $(N) + cols
    DB += group * $(N) + cols
    dw_acc = tl.zeros([$(BLOCK_SIZE_N)], dtype=tl.float32)
    db_acc = tl.zeros([$(BLOCK_SIZE_N)], dtype=tl.float32)
    for row in range(group, $(M), $(GROUP_SIZE_M)) {
      x = tl.load(X + row * $(stride) + cols, mask=mask, other=0.0).to(tl.float32)
      dy = tl.load(DY + row * $(stride) + cols, mask=mask, other=0.0).to(tl.float32)
      mean = tl.load(Mean + row)
      rstd = tl.load(Rstd + row)
      xhat = (x - mean) * rstd
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      wdy = w * dy
      xhat = tl.where(mask, xhat, 0.0)
      wdy = tl.where(mask, wdy, 0.0)
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
      tl.store(DX + row * $(stride) + cols, dx.to(tl.float32), mask=mask)
      partial_dw = (dy * xhat).to(tl.float32)
      partial_db = dy.to(tl.float32)
      if row != group {
        partial_dw += dw_acc
        partial_db += db_acc
      }
      dw_acc = partial_dw.to(tl.float32)
      db_acc = partial_db.to(tl.float32)
    }
    tl.store(DW, dw_acc, mask=mask)
    tl.store(DB, db_acc, mask=mask)
  }


/-- Consume the first reduced vector before forming the second. -/
def finalBackwardCandidate (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel :=
  triton {
    tl.region DW = tl.float32, DB = tl.float32, FINAL_DW = tl.float32, FINAL_DB = tl.float32
    pid = tl.program_id(0)
    cols = pid * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    dw = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    db = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    for i in range(0, $(M), $(BLOCK_SIZE_M)) {
      rows = i + tl.arange(0, $(BLOCK_SIZE_M))
      mask = (rows[:, None] < $(M)) & (cols[None, :] < $(N))
      offs = rows[:, None] * $(N) + cols[None, :]
      dw += tl.load(DW + offs, mask=mask, other=0.0)
      db += tl.load(DB + offs, mask=mask, other=0.0)
    }
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(tl.float32), mask=cols < $(N))
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DB + cols, sum_db.to(tl.float32), mask=cols < $(N))
  }

end VeriTile.Bench.Repairs.LayerNormTriton.TypedSource
