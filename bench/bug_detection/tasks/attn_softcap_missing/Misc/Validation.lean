import bench.bug_detection.tasks.attn_softcap_missing.Correct
import bench.bug_detection.tasks.attn_softcap_missing.FPEquiv

namespace VeriTile.Bench.BugDetection.attn_softcap_missing
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open scoped VeriTile.Triton.LogicalScheduledKernelIO

example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := rfl
example : IO.core.nIn = 3 := rfl
example : IO.core.oarity (0 : Fin 1) = 131072 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.BugDetection.attn_softcap_missing
