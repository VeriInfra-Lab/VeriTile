import bench.tritonbench_g.vector_addition.Proofs.Support.Parameters
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace VeriTile.Bench.Repairs.KldivCompute
abbrev Parameters := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters
end VeriTile.Bench.Repairs.KldivCompute

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n+1023)/1024
noncomputable def mathematical (x y : ℝ) : ℝ := x * Real.log (x/y)
end VeriTile.Bench.Repairs.KldivCompute.Contiguous

namespace VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource
/-- Floating storage labels; native support and discrete conversions are separate obligations. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr
end VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource
