import bench.pretrain_kernels.tasks.local_mtp_roll_packed_dx.Contract

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed (fun xs (i0,i1) =>
    let input0 := xs.input0
    let constant53d3e3c65e := xs.constant53d3e3c65e
    let vslice_backward : Fin 2 → Fin 48 → ℝ := fun i0 i1 => (if (i1.val ≥ 1 ∧ i1.val < 48 ∧ (i1.val - 1) % 1 = 0) then input0 i0 ⟨(((i1.val - 1) + 80)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_1 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val ≥ 80 ∧ i1.val < 128 ∧ (i1.val - 80) % 1 = 0) then vslice_backward i0 ⟨((i1.val - 80)) % 48, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_2 : Fin 2 → Fin 32 → ℝ := fun i0 i1 => (if (i1.val ≥ 1 ∧ i1.val < 32 ∧ (i1.val - 1) % 1 = 0) then input0 i0 ⟨(((i1.val - 1) + 48)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_3 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val ≥ 48 ∧ i1.val < 80 ∧ (i1.val - 48) % 1 = 0) then vslice_backward_2 i0 ⟨((i1.val - 48)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (vslice_backward_1 i0 i1 + vslice_backward_3 i0 i1)
    let vslice_backward_4 : Fin 2 → Fin 32 → ℝ := fun i0 i1 => (if (i1.val ≥ 1 ∧ i1.val < 32 ∧ (i1.val - 1) % 1 = 0) then input0 i0 ⟨(((i1.val - 1) + 16)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_5 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val ≥ 16 ∧ i1.val < 48 ∧ (i1.val - 16) % 1 = 0) then vslice_backward_4 i0 ⟨((i1.val - 16)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_1 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (vadd i0 i1 + vslice_backward_5 i0 i1)
    let vslice_backward_6 : Fin 2 → Fin 16 → ℝ := fun i0 i1 => (if (i1.val ≥ 1 ∧ i1.val < 16 ∧ (i1.val - 1) % 1 = 0) then input0 i0 ⟨((i1.val - 1)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_7 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val ≥ 0 ∧ i1.val < 16 ∧ (i1.val - 0) % 1 = 0) then vslice_backward_6 i0 ⟨((i1.val - 0)) % 16, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_2 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (vadd_1 i0 i1 + vslice_backward_7 i0 i1)
    vadd_2 i0 i1))

end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_dx
