import VeriTile.Triton

/-! Shared original source. The wrapper explicitly selects FP16 or FP32.
Real is the mathematical interpretation; no BF16 branch is introduced. -/

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
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

end VeriTile.Bench.TritonBenchG.MatrixReduction

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
open VeriTile.Triton Lean

inductive Storage where
  | fp16 | fp32
  deriving DecidableEq, Repr

/-- Direct block-pointer source for each original storage profile. -/
def kernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => fun (x_ptr y_ptr : RegionName) (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) =>
    triton {
      tl.region x_ptr = tl.float16, y_ptr = tl.float16
      x_ptr = tl.make_block_ptr(base=x_ptr,
        shape=($(BLOCK_M), $(BLOCK_N)), strides=($(stride_xm), $(stride_xn)),
        offsets=($(0), $(0)), block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
      x = tl.load(x_ptr)
      y = tl.max(x, axis=1)
      tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y.to(tl.float16))
    }
  | .fp32 => fun (x_ptr y_ptr : RegionName) (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) =>
    triton {
      tl.region x_ptr = tl.float32, y_ptr = tl.float32
      x_ptr = tl.make_block_ptr(base=x_ptr,
        shape=($(BLOCK_M), $(BLOCK_N)), strides=($(stride_xm), $(stride_xn)),
        offsets=($(0), $(0)), block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
      x = tl.load(x_ptr)
      y = tl.max(x, axis=1)
      tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y.to(tl.float32))
    }

/-- Direct affine-address candidate source for each original storage profile. -/
def candidateKernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => fun (x_ptr y_ptr : RegionName) (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) =>
    triton {
      tl.region x_ptr = tl.float16, y_ptr = tl.float16
      rows = tl.arange(0, $(BLOCK_M))
      cols = tl.arange(0, $(BLOCK_N))
      offsets = rows[:, None] * $(stride_xm) + cols[None, :] * $(stride_xn)
      x = tl.load(x_ptr + offsets)
      y = tl.max(x, axis=1)
      tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y.to(tl.float16))
    }
  | .fp32 => fun (x_ptr y_ptr : RegionName) (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) =>
    triton {
      tl.region x_ptr = tl.float32, y_ptr = tl.float32
      rows = tl.arange(0, $(BLOCK_M))
      cols = tl.arange(0, $(BLOCK_N))
      offsets = rows[:, None] * $(stride_xm) + cols[None, :] * $(stride_xn)
      x = tl.load(x_ptr + offsets)
      y = tl.max(x, axis=1)
      tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y.to(tl.float32))
    }

/- Materialize the shared equations here so independent Real and FP consumers
reuse the same declarations after source relocation. -/
#guard_msgs (drop info) in
#check kernel.eq_1
#guard_msgs (drop info) in
#check kernel.eq_2

end VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
