import VeriTile.Triton
import VeriTile.Triton.Float.Structural

/-! Historical literal Real source, retained only by explicitly historical
compiler and mathematical helper views. The canonical source is in Kernels. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Triton
/-- Historical literal Real instance, with the same complete three-pass body. -/
def source (dtype : FloatDType) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd stride_y_N stride_y_hn stride_y_hd
      stride_w_hn stride_w_hd N BLOCK_SIZE : Nat) (eps : ℝ) : ComputeKernel := triton {
  tl.region X = DTYPE, W = DTYPE, Y = DTYPE
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols * $(stride_x_hd), mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols * $(stride_x_hd), mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols * $(stride_w_hd), mask=mask).to(tl.float32)
    x = tl.load(X + cols * $(stride_x_hd), mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols * $(stride_y_hd), (y).to(X.dtype.element_ty), mask=mask)
  }
}

/-- Immutable source pin; no original-source equivalence is claimed. -/
def originalSHA256 := "9ad4b038ee7d5fff373705d8ef8c0a4a9a3100a064db07703f89d61b32de6c88"
def candidateSHA256 := "a22618b08165692510fbf836cd2f7ad87943610902d3ee85016824e34e82aae9"
def block : Nat := 128

structure Shape where
  (sequences heads columns : Nat)
  deriving DecidableEq

structure Strides where
  (xSeq xHead xDim ySeq yHead yDim wHead wDim : Nat)
  deriving DecidableEq

def repaired (s : Strides) (N : Nat) (eps : ℝ) : ComputeKernel :=
  source .fp32 "X" "W" "Y" s.xSeq s.xHead s.xDim s.ySeq s.yHead s.yDim
    s.wHead s.wDim N block eps

end VeriTile.Bench.Repairs.LayernormFwdTriton
