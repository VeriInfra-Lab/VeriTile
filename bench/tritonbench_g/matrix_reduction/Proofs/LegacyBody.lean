import VeriTile.Triton

/-! Shared original source. The wrapper explicitly selects FP16 or FP32.
Real is the mathematical interpretation; no BF16 branch is introduced. -/

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.LegacyBody
open VeriTile.Triton

/-- Faithful 1:1 transcription of `matrix_reduction.py`'s `load_reduce_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` → Lean `Nat`
  parameters.
- `stride_y` is kept as `_stride_y`: the upstream Triton kernel accepts it but
  stores to `y_ptr + tl.arange(0, BLOCK_M)` and does not use it. -/
def load_reduce_kernel
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  x_ptr = tl.make_block_ptr(base=x_ptr,
    shape=($(BLOCK_M), $(BLOCK_N)),
    strides=($(stride_xm), $(stride_xn)),
    offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)),
    order=(1, 0))
  x = tl.load(x_ptr)
  y = tl.max(x, axis=1)
  tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y)
}

end VeriTile.Bench.TritonBenchG.MatrixReduction.LegacyBody

