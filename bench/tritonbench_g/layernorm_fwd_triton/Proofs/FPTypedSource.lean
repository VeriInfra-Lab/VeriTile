import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton
/-- Complete original three-pass streaming source. Declared element strides
remain unused exactly as in Python; no single-block specialization is made. -/
def source (dtype : FloatDType)
    (X W Y : RegionName)
    (stride_x_N stride_x_hn _stride_x_hd
      stride_y_N stride_y_hn _stride_y_hd
      stride_w_hn _stride_w_hd : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, Y = DTYPE
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(X.dtype.element_ty), mask=mask)
  }
}
structure Parameters where
  (x weight y : RegionName)
  (xSeqStride xHeadStride xDimStride ySeqStride yHeadStride yDimStride weightHeadStride weightDimStride columns block : Nat)
  epsilon : ℝ

def kernel (p : Parameters) := source .fp32 p.x p.weight p.y
  p.xSeqStride p.xHeadStride p.xDimStride p.ySeqStride p.yHeadStride p.yDimStride
  p.weightHeadStride p.weightDimStride p.columns p.block p.epsilon

def prelude (p : Parameters) := (kernel p).surfaceBody.take 6
def meanPost (p : Parameters) := ((kernel p).surfaceBody.drop 7).take 2
def varPost (p : Parameters) := ((kernel p).surfaceBody.drop 10).take 2

def loopBody (p : Parameters) (index : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel p).surfaceBody
  match code[index]? with
  | some (ComputeStmt.forRange _ _ _ _ xs) => xs
  | _ => []
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
