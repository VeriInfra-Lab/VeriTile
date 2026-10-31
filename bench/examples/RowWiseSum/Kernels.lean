import VeriTile.Triton.DSL.CallSignature

/-! Newly authored Python sources from the independent row-sum specification;
see row_wise_sum.py. Only the reduction's lane order changes. -/
namespace VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32)]
  scalars := []
  constexpr := ["nCol","blockSize"]

def rowWiseSumKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)
  }

def reversedKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    row = tl.program_id(0)
    cols = blockSize - 1 - tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)
  }

end VeriTile.Bench.Examples.RowWiseSum.Kernels
