import bench.tritonbench_g.fp4_to_bf16.Proofs.HostSchedule

/-! Any permutation of the complete original logical grid preserves exact
initial-byte semantics. Program registers are local and memory is threaded
through each actual source execution. Native instruction interleavings,
physical alias/allocator binding and the unrestricted Python dtype/error
domain are outside this contract. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.ScheduleCorrect
open VeriTile.Triton FP.Structural FPValue Host HostSchedule
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The original packed-uint8 to bf16 launch is independent of program order.
Every grid program appears exactly once, with no fixed-order or successful-run
hypothesis. The complete independent reference and its frame are retained. -/
specification fp4_to_bf16_arbitrary_program_order_correctness
    (rest : Algebra Word) (h : Input) (jobs : List Nat) (initial : State Word)
    (fresh : h.input ≠ h.output) (typed : PackedInput h initial)
    (order : jobs.Perm (List.range (grid h))) :
    ∃ final, runSchedule rest h jobs initial = some final ∧
      final.mem = Host.expectedMemory h initial ∧
      (∀ r o, (r ≠ h.output ∨ h.shape.count * 2 ≤ o) →
        final.mem r o = initial.mem r o) := by
  have cover : ∀ pid, pid < grid h → pid ∈ jobs := by
    intro pid hp
    exact order.mem_iff.mpr (List.mem_range.mpr hp)
  obtain ⟨final, hr, hm⟩ := complete_schedule rest h jobs initial fresh typed cover
  refine ⟨final, hr, hm, ?_⟩
  intro r o outside
  rw [hm]
  have inactive : ¬(r = h.output ∧ o < h.shape.count * 2) := by
    rcases outside with hr | ho
    · exact fun active => hr active.1
    · omega
  simp only [Host.expectedMemory, inactive, ↓reduceIte]

#axiomsClean fp4_to_bf16_arbitrary_program_order_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.ScheduleCorrect
