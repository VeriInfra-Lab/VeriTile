import bench.tritonbench_g.fp4_to_bf16.Proofs.FPReal
import bench.tritonbench_g.fp4_to_bf16.Proofs.ScheduleCorrect

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.RealSpec
open VeriTile.Triton FP.Structural FPValue Host HostCorrect HostSchedule
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem read_cell (h : Input) (initial : State Word) (j : Nat) :
    readBf16 (cell h initial j) = ⟨Bits.expectedBits (nibble h initial j)⟩ := by
  simp only [readBf16, cell, value, embed, Cell.read_same, FP.Value.mk.injEq]
  simpa only [BitVec.setWidth_eq] using
    (BitVec.setWidth_setWidth_of_le (Bits.expectedBits (nibble h initial j)) (by decide : 16 ≤ 32))

/-- A complete logical-source execution and frame, projected to the independent
Real formula without erasing the additional exact BF16/signed-zero guarantee.
The finite native index obstruction is separately exposed by IndexOverflow. -/
theorem schedule_real_correctness (rest : Algebra Word) (h : Input)
    (jobs : List Nat) (initial : State Word) (fresh : h.input ≠ h.output)
    (typed : PackedInput h initial) (order : jobs.Perm (List.range (grid h))) :
    ∃ final, runSchedule rest h jobs initial = some final ∧
      (∀ j, j < h.shape.count * 2 →
        (readBf16 (final.mem h.output j)).bits = Bits.expectedBits (nibble h initial j) ∧
        finiteReal (readBf16 (final.mem h.output j)) = some (e2m1 (nibble h initial j))) ∧
      (∀ r o, (r ≠ h.output ∨ h.shape.count * 2 ≤ o) →
        final.mem r o = initial.mem r o) := by
  obtain ⟨final, run, memory, frame⟩ :=
    ScheduleCorrect.fp4_to_bf16_arbitrary_program_order_correctness rest h jobs initial fresh typed order
  refine ⟨final, run, ?_, frame⟩
  intro j hj
  have read : readBf16 (final.mem h.output j) = ⟨Bits.expectedBits (nibble h initial j)⟩ := by
    rw [memory]
    simp only [expectedMemory, hj, and_self, ↓reduceIte]
    exact read_cell h initial j
  rw [read]
  exact ⟨rfl, expected_real _⟩

#axiomsClean schedule_real_correctness
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.RealSpec
