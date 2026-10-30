import bench.bug_detection.tasks.rope_fp16_angles.Contract

namespace VeriTile.Bench.BugDetection.rope_fp16_angles
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (h,r,j) => if j.val < 128 then xs.q h r j * Real.cos (angle r j) + (if j.val < 64 then -xs.q h r (partner j) else xs.q h r (partner j)) * Real.sin (angle r j) else xs.q h r j)

end
end VeriTile.Bench.BugDetection.rope_fp16_angles
