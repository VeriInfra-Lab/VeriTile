import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
open VeriTile.Triton Lean

inductive Storage where
  | fp16 | fp32
  deriving DecidableEq, Repr

syntax "matrixMax%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(matrixMax%[$dtype:tritonDType]) =>
    `(fun (x_ptr y_ptr : RegionName) (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [x_ptr] [y_ptr] (triton {
        tl.region x_ptr = $dtype, y_ptr = $dtype
        x_ptr = tl.make_block_ptr(base=x_ptr,
          shape=($(BLOCK_M), $(BLOCK_N)), strides=($(stride_xm), $(stride_xn)),
          offsets=($(0), $(0)), block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
        x = tl.load(x_ptr)
        y = tl.max(x, axis=1)
        tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y.to($dtype))
      }).surfaceBody)

def kernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => matrixMax%[tl.float16]
  | .fp32 => matrixMax%[tl.float32]

end VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
