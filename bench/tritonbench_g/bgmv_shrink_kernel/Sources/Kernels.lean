import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_Kernels

/-! The checked direct host uses `Direct.Kernels.source` below. The earlier
padded source is retained for archived proof credit. Both reuse the unchanged
complete typed arithmetic: 64 split programs, 256 rank lanes, arbitrary K.
The direct wrapper's checks derive its actual native address bounds. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernel
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource.Storage

structure Call where
  batches : Nat
  adapters : Nat
  N : Nat
  K : Nat
  BN : Nat
  blockPositive : 0 < BN
  rowsCovered : N ≤ BN
  storage : Storage
  outputType : FloatDType
  scaling : ℝ

namespace Call
/-- Python cdiv(K,16384), including zero. -/
def cycles (h : Call) := (h.K+16383)/16384
/-- Actual successful X/W allocations, not an additional shape guard. -/
def paddedK (h : Call) := max 1 h.cycles * 16384
end Call

namespace Kernels
abbrev original := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource.original

def source (h : Call) : ComputeKernel :=
  original h.storage h.outputType "X" "W" "Y" h.N h.K ⟨"Indices"⟩ h.scaling
    h.paddedK 1 (h.BN*h.paddedK) h.paddedK 1 h.BN 1 h.BN 256 64
end Kernels
end VeriTile.Bench.Repairs.BgmvShrinkKernel

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct
open VeriTile.Triton
structure Call where
  batches : Nat
  adapters : Nat
  N : Nat
  K : Nat
  BN : Nat
  columns : Nat
  blockPositive : 0<BN
  rowsCovered : N≤BN
  outputCovered : N≤columns
  storage : Storage
  outputType : FloatDType
  scaling : ℝ
namespace Call
def cycles (h : Call) := (h.K+16383)/16384
def paddedK (h : Call) := max 1 h.cycles*16384
end Call
namespace Kernels
abbrev original := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource.original
/-- Original typed arithmetic on the checked direct contiguous buffers. -/
def source (h : Call) : ComputeKernel :=
  original h.storage h.outputType "X" "W" "Y" h.N h.K ⟨"Indices"⟩ h.scaling
    h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64
end Kernels
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct
