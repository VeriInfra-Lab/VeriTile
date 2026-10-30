import bench.pretrain_kernels.tasks.local_cross_entropy_ignore_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed (fun xs (i0,i1) =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let input0 := xs.input0
    let vge : Fin 128 → Fin 2 → Bool := fun i0 i1 => decide (parameter1 i0 i1 ≥ (1/8))
    let vwhere : Fin 128 → Fin 2 → Int := fun i0 i1 => (if vge i0 i1 then parameter0 i0 i1 else (-100))
    let vwhere_1 : Fin 128 → Fin 2 → Int := fun i0 i1 => (if vge i0 i1 then vwhere i0 i1 else 0)
    let vto_copy : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => input0 i0 i1 i2
    let vmax_1_value : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => ((List.finRange 1024).map (fun lane => vto_copy i0 i1 ⟨(lane.val) % 1024, Nat.mod_lt _ (by decide)⟩)).foldl max (vto_copy i0 i1 ⟨(0) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vmax_1_index : Fin 128 → Fin 2 → Fin 1 → Int := fun i0 i1 i2 => ((((List.finRange 1024).mergeSort (fun left right => decide (vto_copy i0 i1 ⟨(left.val) % 1024, Nat.mod_lt _ (by decide)⟩ > vto_copy i0 i1 ⟨(right.val) % 1024, Nat.mod_lt _ (by decide)⟩ ∨ (vto_copy i0 i1 ⟨(left.val) % 1024, Nat.mod_lt _ (by decide)⟩ = vto_copy i0 i1 ⟨(right.val) % 1024, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).headD 0).val : Int)
    let vsub : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (vto_copy i0 i1 i2 - vmax_1_value i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vexp : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (Real.exp (vsub i0 i1 i2))
    let vsum_1 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (∑ lane : Fin 1024, vexp i0 i1 lane)
    let vlog : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (Real.log (vsum_1 i0 i1))
    let vgather : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => vto_copy i0 i1 ⟨(Int.toNat (vwhere_1 i0 ⟨((i1.val + i2.val)) % 2, Nat.mod_lt _ (by decide)⟩)) % 1024, Nat.mod_lt _ (by decide)⟩
    let vsub_1 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (vgather i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ - vmax_1_value i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsub_2 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (vsub_1 i0 i1 - vlog i0 i1)
    let vneg : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (- (vsub_2 i0 i1))
    let vwhere_2 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (if vge i0 i1 then vneg i0 i1 else 0)
    vwhere_2 i0 i1))

end
end VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_fwd
