import VeriTile.Triton.DSL
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.TypedSource
open VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq
inductive KeepStorage where
  | int32 | bool
  deriving DecidableEq
abbrev sourceType := RegionName → RegionName → RegionName → Nat → Nat → ℝ → ComputeKernel
end VeriTile.Bench.TritonBenchG.DropoutTriton.TypedSource
