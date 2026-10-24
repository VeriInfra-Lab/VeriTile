import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ScalarNative

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RawNative
open VeriTile Triton
noncomputable section

/-- Cells owned by the literal source store geometry across all launches. -/
def Writes (owns : (p : LaunchPlan.Item) → HostFP.Ownership p.size)
    (plans : List LaunchPlan.Item) (region : RegionName) (offset : Nat) : Prop :=
  ∃ p ∈ plans, ∃ i, owns p i region offset

theorem frame (reference : Nat → NativeGrid.State → NativeGrid.State)
    (owns : (p : LaunchPlan.Item) → HostFP.Ownership p.size)
    (plans : List LaunchPlan.Item) (before : NativeGrid.State)
    (region : RegionName) (offset : Nat) (outside : ¬ Writes owns plans region offset) :
    (NativeGrid.referenceBatches reference owns plans before).mem region offset = before.mem region offset := by
  classical
  induction plans generalizing before with
  | nil => rfl
  | cons p ps ih =>
    have head : ¬ ∃ i, owns p i region offset := by
      rintro ⟨i, hi⟩
      exact outside ⟨p, List.mem_cons_self, i, hi⟩
    have tail : ¬ Writes owns ps region offset := by
      rintro ⟨q, hq, i, hi⟩
      exact outside ⟨q, List.mem_cons_of_mem p hq, i, hi⟩
    rw [NativeGrid.referenceBatches, ih _ tail]
    simp [NativeGrid.merge, head]

#axiomsClean frame
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RawNative
