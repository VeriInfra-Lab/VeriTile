import VeriTile.Triton.DSL.CallSignature

/-! Newly authored FP32 Python sources from the independent row maximum
specification; see row_wise_max.py for provenance and launch specialization. -/
namespace VeriTile.Bench.Examples.RowWiseMax.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32)]
  scalars := []
  constexpr := ["nCol","blockSize"]

def rowWiseMaxKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.max(values, axis=0)
    tl.store(yReg + row, result)
  }

def inlinedKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    tl.store(yReg + row, tl.max(tl.load(xReg + row * nCol + cols), axis=0))
  }

end VeriTile.Bench.Examples.RowWiseMax.Kernels
