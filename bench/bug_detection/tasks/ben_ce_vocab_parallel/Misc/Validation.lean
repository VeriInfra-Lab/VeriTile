import bench.bug_detection.tasks.ben_ce_vocab_parallel.Correct
import bench.bug_detection.tasks.ben_ce_vocab_parallel.FPEquiv

namespace VeriTile.Bench.BugDetection.ben_ce_vocab_parallel
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open scoped VeriTile.Triton.LogicalScheduledKernelIO

example : correctnessProblem = Spec.Real (IO.realIO.ImplementsOn Contract.Allowed Contract.output) := rfl
example : IO.core.nIn = 2 := rfl
example : IO.core.oarity (0 : Fin 1) = 64 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.BugDetection.ben_ce_vocab_parallel
