import bench.tritonbench_g.matrix_transpose.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MatrixTranspose

theorem contract_iff (m d : Nat) : InputContract m d ↔
    (0 < m ∧ 0 < d) ∧ (m &&& (m-1) = 0 ∧ d &&& (d-1) = 0) ∧ m*d ≤ 2^20 := by
  constructor
  · intro h; exact ⟨h.positive, h.powers, h.extent⟩
  · rintro ⟨hp, hw, he⟩; exact ⟨hp, hw, he⟩

theorem checkIntegers_iff (m d : Int) : checkIntegers m d = .ok () ↔
    0 < m ∧ 0 < d ∧ InputContract m.toNat d.toNat := by
  rw [contract_iff]
  simp [checkIntegers, ite_eq_iff]
  omega

theorem checkInputs_iff (m d : Argument) :
    checkInputs m d = .ok () ↔ AdmittedArguments m d := by
  cases m <;> cases d <;> simp [checkInputs, AdmittedArguments, checkIntegers_iff]

theorem check_nat_iff (m d : Nat) :
    checkInputs (.integer m) (.integer d) = .ok () ↔ InputContract m d := by
  simp [checkInputs, checkIntegers_iff, contract_iff]
  tauto

/-- This guard admits every frontend-legal original natural shape. -/
theorem contract_iff_sourceShape (m d : Nat) :
    InputContract m d ↔ SourceShape m d := by
  rw [contract_iff]
  constructor
  · rintro ⟨⟨hm, hd⟩, ⟨hmp, hdp⟩, he⟩
    obtain ⟨a, ha⟩ := (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).1 hmp
    obtain ⟨b, hb⟩ := (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).1 hdp
    refine ⟨a, b, ha, hb, ?_⟩
    rw [ha, hb, ← Nat.pow_add] at he
    exact (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).1 he
  · rintro ⟨a, b, rfl, rfl, he⟩
    refine ⟨⟨Nat.two_pow_pos a, Nat.two_pow_pos b⟩, ?_, ?_⟩
    · exact ⟨(Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by have := Nat.two_pow_pos a; omega)).2 ⟨a, rfl⟩,
        (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by have := Nat.two_pow_pos b; omega)).2 ⟨b, rfl⟩⟩
    · rw [← Nat.pow_add]; exact Nat.pow_le_pow_right (by decide) he

#axiomsClean checkInputs_iff
#axiomsClean check_nat_iff
#axiomsClean contract_iff_sourceShape
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatrixTranspose
