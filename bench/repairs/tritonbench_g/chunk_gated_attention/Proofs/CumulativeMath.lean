import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Host

/-! Independent logical chunk prefix, equivalent finite-sum forms. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000

theorem prefix_range (BT c i f : Nat) (a : Nat → Nat → ℝ) (hi : i < BT) :
    Host.chunkPrefix BT a (c*BT+i) f = ∑ u ∈ Finset.range (i+1), a (c*BT+u) f := by
  have hBT : 0 < BT := Nat.zero_lt_of_lt hi
  have hd : (c*BT+i)/BT = c := by
    simp [Nat.add_comm _ i,Nat.add_mul_div_right _ _ hBT,Nat.div_eq_of_lt hi]
  unfold Host.chunkPrefix
  rw [hd,show c*BT+i+1=c*BT+(i+1) by omega,Finset.sum_range_add]
  have hz : (∑ u ∈ Finset.range (c*BT), if c*BT ≤ u then a u f else 0) = 0 := by
    apply Finset.sum_eq_zero
    intro u hu
    simp only [Finset.mem_range] at hu
    simp [Nat.not_le_of_lt hu]
  rw [hz,zero_add]
  apply Finset.sum_congr rfl
  intro u hu
  simp

theorem prefix_tile (BT c f : Nat) (i : Fin BT) (a : Nat → Nat → ℝ) :
    (∑ u ∈ Finset.univ.filter (fun u : Fin BT => u.val ≤ i.val), a (c*BT+u.val) f) =
      Host.chunkPrefix BT a (c*BT+i.val) f := by
  rw [prefix_range _ _ _ _ _ i.isLt,Finset.sum_filter]
  rw [← Finset.sum_range (fun u : Nat => if u ≤ i.val then a (c*BT+u) f else 0)]
  have hsub : Finset.range (i.val+1) ⊆ Finset.range BT := Finset.range_mono (Nat.succ_le_iff.mpr i.isLt)
  rw [← Finset.sum_subset hsub (by
    intro u hu hn
    simp only [Finset.mem_range] at hn
    simp [show ¬u ≤ i.val by omega])]
  apply Finset.sum_congr rfl
  intro u hu
  simp only [Finset.mem_range] at hu
  simp [show u ≤ i.val by omega]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeMath
