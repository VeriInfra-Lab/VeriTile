import VeriTile.Triton.DSL.CallSignature

/-! The registered Python companion supplies the complete function bodies.
The original add_kernel body is unchanged. The comparison commutes only its
addition. Invocation bindings and FP32 storage types are outside the bodies. -/
namespace VeriTile.Bench.Examples.VectorAdd.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["BLOCK_SIZE"]

def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
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
    tl.store(out_ptr + offsets, output)
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
    tl.store(out_ptr + offsets, output)
  }

def originalKernel (blockSize : Nat) : ComputeKernel :=
  addKernel "x" "y" "out" blockSize

def optimizedKernel (blockSize : Nat) : ComputeKernel :=
  commutedKernel "x" "y" "out" blockSize

end VeriTile.Bench.Examples.VectorAdd.Kernels
