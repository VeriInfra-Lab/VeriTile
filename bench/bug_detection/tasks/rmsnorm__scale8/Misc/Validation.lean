import bench.bug_detection.tasks.rmsnorm__scale8.Correct
import bench.bug_detection.tasks.rmsnorm__scale8.FPEquiv

namespace VeriTile.Bench.BugDetection.rmsnorm__scale8
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open scoped VeriTile.Triton.LogicalScheduledKernelIO

example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := rfl
example : IO.core.nIn = 2 := rfl
example : IO.core.oarity (0 : Fin 1) = 32768 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.BugDetection.rmsnorm__scale8
