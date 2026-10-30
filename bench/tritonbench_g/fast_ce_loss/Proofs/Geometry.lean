import bench.tritonbench_g.fast_ce_loss.Proofs.Source

namespace VeriTile.Bench.Repairs.FastCeLoss.Geometry
open Source
noncomputable section
set_option maxHeartbeats 2000000

theorem ceil_cover (n block : Nat) (positive : 0<block) : n≤block*ceilDiv n block := by
  have split := Nat.mod_add_div n block
  have remainder := Nat.mod_lt n positive
  unfold ceilDiv
  split_ifs with h
  · simp only [Nat.add_zero]
    omega
  · simp only [Nat.mul_add,Nat.mul_one]
    omega

theorem owner_bound {n block col : Nat} (positive : 0<block) (active : col<n) :
    col/block<ceilDiv n block := by
  apply (Nat.div_lt_iff_lt_mul positive).mpr
  simpa only [Nat.mul_comm] using lt_of_lt_of_le active (ceil_cover n block positive)

theorem ceil_positive (n block : Nat) (positive : 0<block) (nonempty : 0<n) :
    0<ceilDiv n block := by
  exact lt_of_le_of_lt (Nat.zero_le _) (owner_bound positive nonempty)

theorem chunk_starts {n block chunk : Nat} (positive : 0<block)
    (active : chunk<ceilDiv n block) : chunk*block<n := by
  have split := Nat.mod_add_div n block
  have remainder := Nat.mod_lt n positive
  unfold ceilDiv at active
  split_ifs at active with h
  · have bound := Nat.mul_le_mul_right block (Nat.succ_le_of_lt active)
    simp only [Nat.succ_mul] at bound
    nlinarith
  · have bound := Nat.mul_le_mul_right block (Nat.le_of_lt_succ active)
    have remPositive : 0<n%block := by omega
    nlinarith

theorem owner_unique {col block q : Nat} (positive : 0<block)
    (lower : q*block≤col) (upper : col<(q+1)*block) : col/block=q := by
  apply Nat.le_antisymm
  · have h : col/block<q+1 := (Nat.div_lt_iff_lt_mul positive).mpr (by simpa [Nat.mul_comm] using upper)
    omega
  · exact (Nat.le_div_iff_mul_le positive).mpr (by simpa [Nat.mul_comm] using lower)

/-- Every logical column belongs to exactly one launched backward block. -/
theorem backward_owner (s : Shape) (col : Nat) (active : col<s.vocab) :
    col/4096<s.blocks ∧ col/4096*4096≤col ∧ col<(col/4096+1)*4096 := by
  have lower := Nat.div_mul_le_self col 4096
  have split := Nat.mod_add_div col 4096
  have remainder := Nat.mod_lt col (by decide : 0<4096)
  exact ⟨owner_bound (by decide) active,lower,by omega⟩

#axiomsClean ceil_cover
#axiomsClean owner_bound
#axiomsClean chunk_starts
#axiomsClean backward_owner
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Geometry
