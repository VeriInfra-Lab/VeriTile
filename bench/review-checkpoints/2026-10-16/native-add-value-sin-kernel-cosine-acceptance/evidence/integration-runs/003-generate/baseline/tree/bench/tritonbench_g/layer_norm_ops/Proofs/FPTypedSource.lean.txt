import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton
/-! Task-local fp32 storage transcriptions. Explicit same-fp32 load casts
materialize the computational carrier; store casts materialize the native
pointer conversion. Single-row DW/DB retain Python's +0 accumulator and add,
which the old mathematical helper elided. The old declarations are unchanged. -/
def meanStore (dtype : FloatDType)
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  tl.region MeanPre = DTYPE, Mean = DTYPE
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}

def rstdStore (dtype : FloatDType)
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  tl.region RstdPre = DTYPE, Rstd = DTYPE
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}

def vectorStore (dtype : FloatDType)
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region ValuePre = DTYPE, Out = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}

def rmsRow (dtype : FloatDType)
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, DY = DTYPE, DX = DTYPE, DW = DTYPE, Rstd = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0).to(tl.float32)
  rstd = tl.load(Rstd + row).to(tl.float32)
  xhat = x * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw += dy * xhat
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, (dx).to(DTYPE), mask=mask)
  tl.store(DW + row * $(N) + cols, (dw).to(DTYPE), mask=mask)
}

def plainRow (dtype : FloatDType)
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, DY = DTYPE, DX = DTYPE, DW = DTYPE, DB = DTYPE, Mean = DTYPE, Rstd = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  db = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0).to(tl.float32)
  mean = tl.load(Mean + row).to(tl.float32)
  rstd = tl.load(Rstd + row).to(tl.float32)
  xhat = (x - mean) * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw += dy * xhat
  db += dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  c2 = tl.sum(wdy, axis=0) / $(N)
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, (dx).to(DTYPE), mask=mask)
  tl.store(DW + row * $(N) + cols, (dw).to(DTYPE), mask=mask)
  tl.store(DB + row * $(N) + cols, (db).to(DTYPE), mask=mask)
}

def residualAdd (dtype : FloatDType)
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region DXBase = DTYPE, DRESIDUAL = DTYPE, DX = DTYPE, DRESIDUAL_IN = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dx = tl.load(DXBase + row * $(stride_dx_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dres = tl.load(DRESIDUAL + row * $(stride_dres_row) + cols, mask=mask, other=0.0).to(tl.float32)
  out = dx + dres
  tl.store(DRESIDUAL_IN + row * $(stride_dres_in_row) + cols, (out).to(DTYPE), mask=mask)
  tl.store(DX + row * $(stride_dx_row) + cols, (out).to(DTYPE), mask=mask)
}

def c1Stage (dtype : FloatDType)
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region Xhat = DTYPE, W = DTYPE, DY = DTYPE, C1 = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0).to(tl.float32)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, (c1).to(DTYPE))
}

def c2Stage (dtype : FloatDType)
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region W = DTYPE, DY = DTYPE, C2 = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c2 = tl.sum(wdy, axis=0) / $(N)
  tl.store(C2 + row, (c2).to(DTYPE))
}

def rmsDX (dtype : FloatDType)
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region Xhat = DTYPE, W = DTYPE, DY = DTYPE, Rstd = DTYPE, C1 = DTYPE, DX = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0).to(tl.float32)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row).to(tl.float32)
  c1 = tl.load(C1 + row).to(tl.float32)
  wdy = w * dy
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, (dx).to(DTYPE), mask=mask)
}

def plainDX (dtype : FloatDType)
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  tl.region Xhat = DTYPE, W = DTYPE, DY = DTYPE, Rstd = DTYPE, C1 = DTYPE, C2 = DTYPE, DX = DTYPE
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0).to(tl.float32)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row).to(tl.float32)
  c1 = tl.load(C1 + row).to(tl.float32)
  c2 = tl.load(C2 + row).to(tl.float32)
  wdy = w * dy
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, (dx).to(DTYPE), mask=mask)
}

structure Parameters where
  (x xhat weight dy dx dw db mean rstd c1 c2 dxBase dres dresIn : RegionName)
  (xStride xhatStride dyStride dxStride dresStride dresInStride columns block : Nat)

def rmsKernel (p : Parameters) := rmsRow .fp32 p.x p.weight p.dy p.dx p.dw p.rstd
  p.xStride p.dyStride p.dxStride p.columns p.block
def plainKernel (p : Parameters) := plainRow .fp32 p.x p.weight p.dy p.dx p.dw p.db p.mean p.rstd
  p.xStride p.dyStride p.dxStride p.columns p.block
def c1Kernel (p : Parameters) := c1Stage .fp32 p.xhat p.weight p.dy p.c1
  p.xhatStride p.dyStride p.columns p.block
def c2Kernel (p : Parameters) := c2Stage .fp32 p.weight p.dy p.c2 p.dyStride p.columns p.block
def rmsDXKernel (p : Parameters) := rmsDX .fp32 p.xhat p.weight p.dy p.rstd p.c1 p.dx
  p.xhatStride p.dyStride p.dxStride p.columns p.block
def plainDXKernel (p : Parameters) := plainDX .fp32 p.xhat p.weight p.dy p.rstd p.c1 p.c2 p.dx
  p.xhatStride p.dyStride p.dxStride p.columns p.block
def residualKernel (p : Parameters) := residualAdd .fp32 p.dxBase p.dres p.dx p.dresIn
  p.dxStride p.dresStride p.dresInStride p.columns p.block
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
