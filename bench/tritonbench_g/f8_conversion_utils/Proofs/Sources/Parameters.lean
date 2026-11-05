import VeriTile.Triton.DSL.CallSignature

/-! Storage choices shared by the original conversion bodies and their
proof interfaces. No source statements or execution are defined here. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
open VeriTile.Triton
inductive Format where
  | e4 | e5
  deriving DecidableEq
inductive Input where
  | fp16 | fp32
  deriving DecidableEq
abbrev sourceType := RegionName → RegionName → Nat → Nat → ComputeKernel
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
