import bench.tritonbench_g.vector_addition.Parameters
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
