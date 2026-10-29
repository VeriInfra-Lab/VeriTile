import bench.pretrain_kernels.tasks.local_triton_topk.Contract

namespace VeriTile.Bench.PretrainKernels.local_triton_topk
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
    let literal_nan : Fin 1 → ℝ := fun _ => 0
    let visnan : Fin 256 → Fin 4096 → Bool := fun i0 i1 => decide (False)
    let vwhere : Fin 256 → Fin 4096 → ℝ := fun i0 i1 => (if visnan i0 i1 then literal_nan ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else input0 i0 i1)
    let vwhereFinite : Fin 256 → Fin 4096 → Prop := fun i0 i1 => (if visnan i0 i1 then False else True)
    let vsort_value : Fin 256 → Fin 4096 → ℝ := fun i0 i1 => vwhere i0 ⟨((((List.finRange 4096).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 4096, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 4096, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 4096, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 4096, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 4096, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 256 → Fin 4096 → Int := fun i0 i1 => ((((List.finRange 4096).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 4096, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 4096, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 4096, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 4096, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vgather : Fin 256 → Fin 8 → ℝ := fun i0 i1 => input0 i0 ⟨(Int.toNat (vsort_index i0 ⟨(i1.val) % 4096, Nat.mod_lt _ (by decide)⟩)) % 4096, Nat.mod_lt _ (by decide)⟩
    vgather i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_triton_topk
