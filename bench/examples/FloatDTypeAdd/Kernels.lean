import VeriTile.Triton.DSL.CallSignature

/-! Newly authored Python sources implement the independent FP32 addition
specification; see float_dtype_add.py. The explicit output cast and addition
order are preserved verbatim in each source. -/
namespace VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["BLOCK_SIZE"]

def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  let x_ptr := xReg
  let y_ptr := yReg
  let out_ptr := outReg
  let BLOCK_SIZE := blockSize
  with_triton_signature callSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(out_ptr + offsets, output.to(tl.float32))
  }

def commutedKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  let x_ptr := xReg
  let y_ptr := yReg
  let out_ptr := outReg
  let BLOCK_SIZE := blockSize
  with_triton_signature callSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = y + x
    tl.store(out_ptr + offsets, output.to(tl.float32))
  }

def originalKernel (blockSize : Nat) : ComputeKernel :=
  floatAddKernel "x" "y" "out" blockSize

def optimizedKernel (blockSize : Nat) : ComputeKernel :=
  commutedKernel "x" "y" "out" blockSize

end VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
