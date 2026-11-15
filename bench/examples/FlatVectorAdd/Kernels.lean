import VeriTile.Triton.DSL.CallSignature

/-! Python sources newly authored from the independent masked-add
specification; see flat_vector_add.py. Shape specialization and FP32 pointer
types stay outside the pasted function bodies. -/
namespace VeriTile.Bench.Examples.FlatVectorAdd.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32),("outReg",.fp32)]
  scalars := []
  constexpr := ["blockSize","nElements"]

def flat_vector_addKernel (xReg yReg outReg : RegionName) (blockSize nElements : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(xReg + offsets, mask=mask)
    y = tl.load(yReg + offsets, mask=mask)
    output = x + y
    tl.store(outReg + offsets, output, mask=mask)
  }

def flat_vector_add_commutedKernel (xReg yReg outReg : RegionName) (blockSize nElements : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    pid = tl.program_id(axis=0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(xReg + offsets, mask=mask)
    y = tl.load(yReg + offsets, mask=mask)
    output = y + x
    tl.store(outReg + offsets, output, mask=mask)
  }

def originalKernel (nElements blockSize : Nat) : ComputeKernel :=
  flat_vector_addKernel "x" "y" "out" blockSize nElements

def optimizedKernel (nElements blockSize : Nat) : ComputeKernel :=
  flat_vector_add_commutedKernel "x" "y" "out" blockSize nElements

end VeriTile.Bench.Examples.FlatVectorAdd.Kernels
