import bench.repairs.tritonbench_g.mean_reduction.Kernels
import bench.repairs.tritonbench_g.mean_reduction.Contract
import bench.tritonbench_g.mean_reduction.Proofs.WholeFPFamily

namespace VeriTile.Bench.Repairs.MeanReduction.KernelFP
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MeanReduction
noncomputable section

theorem source_is_shared (M N : Nat) : WholeFPValue.source "X" "Mean" M N = kernel M N := rfl

/-- Both complete actual source variants succeed to the same complete state.
 This retains FP32 load, casts, every streaming add, reduction and division. -/
theorem variants_execute {α : Type} [Inhabited α] (A : Algebra α)
    (M N : Nat) (s : State α) :
    ∃ t, exec A (kernel M N) s = some t ∧
      exec A (indexCommutedKernel M N) s = some t := by
  obtain ⟨t, he, _, _⟩ := WholeFPValue.tile_result A "X" "Mean" M N s
  exact ⟨t, he, (WholeFP.tile_equal A "X" "Mean" M N s).symm.trans he⟩

/-- Typed kernel input is the original source's compressed buffer. The core
 laws bind its observed metadata and data pointer to the same physical storage.
 This permits every FP32 word, without a finite-value requirement. -/
def InputCells (x : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws x) (contiguous : x.metadata.isContiguous = Bool.true)
    (read : Nat → VeriTile.Torch.FP32.Word) (s : State VeriTile.Torch.FP32.Word) : Prop :=
  ∀ i, i < x.metadata.numel → s.mem "X" i = .mk .real
    (read ((VeriTile.Torch.FP32.dataView x laws contiguous).address i))

/-- Full source-grid success, exact primitive formula, and complete memory
 frame. The algebra may vary by program to retain the actual reduction tree. -/
theorem grid {α : Type} [Inhabited α] (family : Nat → Algebra α)
    (M N : Nat) (s : State α) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (Whole.gridSize M))) :
    ∃ t, WholeFPFamily.runTiles family Bool.false "X" "Mean" M N tiles s = some t ∧
      WholeFPFamily.runTiles family Bool.true "X" "Mean" M N tiles s = some t ∧
      (∀ i, i < M → t.mem "Mean" i =
        .mk .real (WholeFPFamily.rowValue family "X" M N s.mem i)) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := WholeFPFamily.grid_result family "X" "Mean" M N
    (by decide) tiles schedule s
  exact ⟨t, he, (WholeFPFamily.grid_equal family "X" "Mean" M N tiles s).symm.trans he, hv, hf⟩

#axiomsClean source_is_shared
#axiomsClean variants_execute
#axiomsClean grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.KernelFP
