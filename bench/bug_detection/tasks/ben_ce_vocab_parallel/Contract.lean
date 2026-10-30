import bench.bug_detection.tasks.ben_ce_vocab_parallel.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_ce_vocab_parallel
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical


def Allowed (xs : IO.Inputs) : Prop := ∀ r, xs.target r < 1024

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs r => Real.log (∑ j : Fin 1024, Real.exp (xs.x r j)) - xs.x r ⟨xs.target r % 1024, Nat.mod_lt _ (by decide)⟩

end Contract
end
end VeriTile.Bench.BugDetection.ben_ce_vocab_parallel
