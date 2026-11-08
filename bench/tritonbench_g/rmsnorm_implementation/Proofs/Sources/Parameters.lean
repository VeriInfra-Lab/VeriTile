import VeriTile.Triton

/-! Storage choices and independent host layout data shared by the selected
source and its existing adapters. No program execution is defined here. -/
namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton
/-- Independent input, weight and output storage; the wrapper uses output=input. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ℝ → ComputeKernel

namespace Whole
abbrev Memory (α : Type) := Nat → Nat → α

structure Tensor where
  shape : List Nat
  stride : Nat → Nat
  allocation : Nat
  base : Nat
  device : Nat
  dtype : Storage

structure Parameters where
  batches : Nat
  rows : Nat
  width : Nat
  input : Tensor
  weights : Tensor
  output : Tensor
  eps : ℝ

end Whole
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
