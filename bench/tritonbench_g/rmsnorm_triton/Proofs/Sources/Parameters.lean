import VeriTile.Triton

/-! Independent storage and tensor layout data for the source and its IO. -/
namespace VeriTile.Bench.TritonBenchG.RmsnormTriton
open VeriTile.Triton
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

def inputOffset (p : Parameters) (b m k : Nat) : Nat :=
  p.input.base + b*p.input.stride 0 + m*p.input.stride 1 + k*p.input.stride 2

def weightOffset (p : Parameters) (k : Nat) : Nat :=
  p.weights.base + k*p.weights.stride 0

def outputOffset (p : Parameters) (b m k : Nat) : Nat :=
  p.output.base + b*p.output.stride 0 + m*p.output.stride 1 + k*p.output.stride 2

end Whole
end VeriTile.Bench.TritonBenchG.RmsnormTriton
