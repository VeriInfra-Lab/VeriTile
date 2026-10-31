import VeriTile.Triton.DSL

/-! Shared storage and call types for the repaired persistent source. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource
open VeriTile.Triton

inductive Storage where
  | fp16 | f8e4
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Nat → Nat → Nat → Nat → Nat → ComputeKernel

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource
