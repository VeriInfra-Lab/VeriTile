import bench.bug_detection.tasks.ben_ce_fused.Contract

namespace VeriTile.Bench.BugDetection.ben_ce_fused
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed ( fun xs r => Real.log (∑ j : Fin 1024, Real.exp (xs.x r j)) - xs.x r ⟨xs.target r % 1024, Nat.mod_lt _ (by decide)⟩))

end
end VeriTile.Bench.BugDetection.ben_ce_fused
