import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeStaging

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeHost
open VeriTile Triton FP.Structural
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
noncomputable section

theorem small_power_table : ∀ k : Fin 257, k.val≠0 →
    ∃ e : Fin 9, Host.nextPowerOfTwo k.val=2^e.val := by decide

theorem direct_profile (c : Host.Dimensions) (hk : c.K≤256) :
    ∃ e : Fin 9, c.intraBK=2^e.val :=
  small_power_table ⟨c.K,by omega⟩ (Nat.ne_of_gt c.k_pos)

theorem inter_small_table : ∀ k : Fin 65,
    min 64 (Host.nextPowerOfTwo (max 16 k.val))=16 ∨
    min 64 (Host.nextPowerOfTwo (max 16 k.val))=32 ∨
    min 64 (Host.nextPowerOfTwo (max 16 k.val))=64 := by decide

theorem inter_profile (c : Host.Dimensions) : c.interBK=16 ∨ c.interBK=32 ∨ c.interBK=64 := by
  by_cases hk : c.K≤64
  · exact inter_small_table ⟨c.K,by omega⟩
  · right; right
    have hp := Host.nextPowerOfTwo_ge (max 16 c.K)
    have hh := Nat.le_max_right 16 c.K
    exact min_eq_left (by omega)

theorem output_small_table : ∀ k : Fin 33,
    min 32 (Host.nextPowerOfTwo (max 16 k.val))=16 ∨
    min 32 (Host.nextPowerOfTwo (max 16 k.val))=32 := by decide

theorem output_profile (c : Host.Dimensions) : c.outputBK=16 ∨ c.outputBK=32 := by
  by_cases hk : c.K≤32
  · exact output_small_table ⟨c.K,by omega⟩
  · right
    have hp := Host.nextPowerOfTwo_ge (max 16 c.K)
    have hh := Nat.le_max_right 16 c.K
    exact min_eq_left (by omega)

theorem value_small_table : ∀ v : Fin 33, ∃ e : Fin 6,
    min 32 (Host.nextPowerOfTwo (max 1 v.val))=2^e.val := by decide

theorem value_profile (c : Host.Dimensions) : ∃ e : Fin 6, c.BV=2^e.val := by
  by_cases hv : c.V≤32
  · exact value_small_table ⟨c.V,by omega⟩
  · refine ⟨⟨5,by decide⟩,?_⟩
    have hp := Host.nextPowerOfTwo_ge (max 1 c.V)
    have hh := Nat.le_max_right 1 c.V
    exact min_eq_left (by omega)

/-- The complete K width of every source dot is an exact instruction block
multiple; source masks handle logical K/T/V tails within those tiles. -/
theorem dot_widths (c : Host.Dimensions) : 8∣c.interBK ∧ 8∣c.outputBK ∧ 8∣c.BT := by
  refine ⟨?_,?_,dvd_trans (by decide : 8∣16) c.bt_divisible⟩
  · rcases inter_profile c with h|h|h <;> rw [h] <;> decide
  · rcases output_profile c with h|h <;> rw [h] <;> decide

theorem reduction_profile_coverage (c : Host.Dimensions) (w : Fin 4) (hk : c.K≤256) :
    (NativeStaging.planFor c.intraBK w).leaves.Perm (List.range c.intraBK) := by
  obtain ⟨e,he⟩ := direct_profile c hk
  rw [he]
  exact NativeStaging.reduction_leaf_coverage e w

theorem split_profile_coverage (w : Fin 4) :
    (NativeStaging.planFor 128 w).leaves.Perm (List.range 128) :=
  NativeStaging.reduction_leaf_coverage ⟨7,by decide⟩ w

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeHost
