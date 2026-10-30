import bench.bug_detection.tasks.rope_fp16_angles.Correct
import bench.bug_detection.tasks.rope_fp16_angles.FPEquiv

namespace VeriTile.Bench.BugDetection.rope_fp16_angles
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open scoped VeriTile.Triton.LogicalScheduledKernelIO

example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := rfl
example : IO.core.nIn = 1 := rfl
example : IO.core.oarity (0 : Fin 1) = 131072 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.BugDetection.rope_fp16_angles
