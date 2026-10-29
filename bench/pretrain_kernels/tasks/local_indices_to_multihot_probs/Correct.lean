import bench.pretrain_kernels.tasks.local_indices_to_multihot_probs.Contract

namespace VeriTile.Bench.PretrainKernels.local_indices_to_multihot_probs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1) =>
    let input0 := xs.input0
    let input1 := xs.input1
    let vne : Fin 128 → Fin 4 → Bool := fun i0 i1 => decide (input0 i0 i1 ≠ (-1))
    let vlt : Fin 128 → Fin 4 → Bool := fun i0 i1 => decide (input0 i0 i1 < 16)
    let vbitwise_and : Fin 128 → Fin 4 → Bool := fun i0 i1 => (vne i0 i1 && vlt i0 i1)
    let vwhere : Fin 128 → Fin 4 → Int := fun i0 i1 => (if vbitwise_and i0 i1 then input0 i0 i1 else 16)
    let vscatter_1 : Fin 128 → Fin 17 → ℝ := fun i0 i1 => ((List.finRange 4).foldl (fun value lane => if vwhere i0 ⟨(lane.val) % 4, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then input1 i0 ⟨(lane.val) % 4, Nat.mod_lt _ (by decide)⟩ else value) (0))
    vscatter_1 i0 ⟨(i1.val) % 17, Nat.mod_lt _ (by decide)⟩)

end
end VeriTile.Bench.PretrainKernels.local_indices_to_multihot_probs
