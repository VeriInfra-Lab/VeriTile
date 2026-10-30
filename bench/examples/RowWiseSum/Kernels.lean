import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! New spec-derived native sources; provenance and mathematical source pins
are recorded in source-references.json. The raw bodies match native_kernels.py. -/

namespace VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32), ("yReg",.fp32)]
  scalars := [("nCol",.i64)]
  constexpr := ["blockSize"]

def rowWiseSumKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [nCol := nCol] in
  triton {
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)
  }

def reversedKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [nCol := nCol] in
  triton {
    row = tl.program_id(0)
    cols = (blockSize - 1) - tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)
  }

end VeriTile.Bench.Examples.RowWiseSum.Kernels
