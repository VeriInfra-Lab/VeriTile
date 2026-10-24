import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
open VeriTile.Triton
/-- Complete original forward body. Storage declarations retain typed cells;
the denominator conversion materializes Python's implicit fp32 scalar
conversion. Reductions retain their input type and padded variance terms. -/
def source (dtype : FloatDType)
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Y_ptr = DTYPE, X_ptr = DTYPE, W_ptr = DTYPE, B_ptr = DTYPE, Mean_ptr = DTYPE, RSTD_ptr = DTYPE
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y_ptr += row_idx * $(Y_row_stride)
  X_ptr += row_idx * $(X_row_stride)
  Mean_ptr += row_idx * $(Mean_row_stride)
  RSTD_ptr += row_idx * $(RSTD_row_stride)
  X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
  W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
  B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
  mean = tl.sum(X_row, axis=0) / ($((n_cols : ℝ))).to(tl.float32)
  var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / ($((n_cols : ℝ))).to(tl.float32)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean_ptr, (mean).to(DTYPE))
  tl.store(RSTD_ptr, (rstd).to(DTYPE))
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y_ptr + col_offsets, (Y_row).to(DTYPE), mask=mask)
}

structure Parameters where
  (y x weight bias mean rstd : RegionName)
  (yStride xStride weightStride biasStride meanStride rstdStride columns block : Nat)
  epsilon : ℝ

def kernel (p : Parameters) : ComputeKernel :=
  source .fp32 p.y p.yStride p.x p.xStride p.weight p.weightStride p.bias p.biasStride
    p.mean p.meanStride p.rstd p.rstdStride p.columns p.epsilon p.block

end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
