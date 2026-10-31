import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_Geometry

/-! Storage and accumulation labels used by the independent signed-loop model. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
open Lean VeriTile.Triton
set_option linter.unusedVariables false

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Pinned Triton exp admits fp32/fp64; other output dtypes are unsupported source profiles. -/
inductive Accumulation where
  | fp32 | fp64
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → Nat → Nat → Nat → ComputeKernel

end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton

end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
