import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! New spec-derived native sources; provenance and mathematical source pins
are recorded in source-references.json. The raw bodies match native_kernels.py. -/

namespace VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32), ("yReg",.fp32), ("outReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def candidateSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    y = tl.load(yReg + offs)
    out = x + y
    tl.store(outReg + offs, out.to(tl.float32))
  }

def optimizedKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let y_ptr : RegionName := "y"
  let out_ptr : RegionName := "out"
  with_triton_signature candidateSignature in
  triton {
    pid = tl.program_id(axis=0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    out = y + x
    tl.store(out_ptr + offs, out.to(tl.float32))
  }

def originalKernel (blockSize : Nat) : ComputeKernel :=
  floatAddKernel "x" "y" "out" blockSize

end VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
