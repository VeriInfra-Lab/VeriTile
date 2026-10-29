import bench.pretrain_kernels.tasks.local_muon_fused_vs_loop.Correct
import bench.pretrain_kernels.tasks.local_muon_fused_vs_loop.FPEquiv

namespace VeriTile.Bench.PretrainKernels.local_muon_fused_vs_loop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped VeriTile.Triton.LogicalScheduledKernelIO
/-- Let the kernel check Eq.refl directly, preserving the long mathematical
let-chain instead of eagerly normalizing it in the tactic unifier. -/
example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := by
  run_tac do
    let goal ← Lean.Elab.Tactic.getMainGoal
    goal.assign (← Lean.Meta.mkEqRefl (Lean.mkConst ``correctnessProblem))
example : IO.core.nIn = 2 := rfl
example : IO.core.oarity (0 : Fin 1) = 107712 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.PretrainKernels.local_muon_fused_vs_loop
