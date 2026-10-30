import bench.bug_detection.tasks.rope__sq16.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.rope__sq16
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def angle (r : Fin 16) (j : Fin 128) : ℝ :=
  (r.val : ℝ) * Real.rpow 10000 (-(2 * ((j.val % 64 : Nat) : ℝ)) / 128)
def partner (j : Fin 128) : Fin 128 :=
  ⟨(if j.val < 64 then j.val + 64 else j.val - 64) % 128, Nat.mod_lt _ (by decide)⟩

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (h,r,j) => if j.val < 128 then xs.q h r j * Real.cos (angle r j) + (if j.val < 64 then -xs.q h r (partner j) else xs.q h r (partner j)) * Real.sin (angle r j) else xs.q h r j

end Contract
end
end VeriTile.Bench.BugDetection.rope__sq16
