import bench.bug_detection.tasks.ben_embedding_onehot.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_embedding_onehot
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def index (xs : IO.Inputs) (r : Fin 64) : Fin 1024 :=
  ((List.finRange 1024).mergeSort (fun a b => decide (xs.x r a > xs.x r b ∨ (xs.x r a = xs.x r b ∧ a.val ≤ b.val)))).headD 0

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,h) => xs.weight (index xs r) h

end Contract
end
end VeriTile.Bench.BugDetection.ben_embedding_onehot
