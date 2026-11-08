import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.Sources.Mathematical_Kernels

/-! Shared geometry records for mathematical adapters and the direct call.
The padded geometry record supports existing mathematical lemmas; no alternative
executable implementation is defined here. The task-level `Kernels.lean` owns
both complete source quotations for the direct repaired launch. -/
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

end Kernels
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct
