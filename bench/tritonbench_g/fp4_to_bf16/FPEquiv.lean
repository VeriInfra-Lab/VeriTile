import bench.tritonbench_g.fp4_to_bf16.Kernels
import bench.tritonbench_g.fp4_to_bf16.Proofs.ScheduleCorrect
import bench.tritonbench_g.fp4_to_bf16.Proofs.IndexOverflow

/-! Exact logical FP equivalence for arbitrary complete schedules and arbitrary
unused backend operations. The selected profile remains uint8 → signed32
payload → FP32 bitcast → BF16. No numerical atom is deferred. The original
finite-index wrapper defect is exposed separately; this is not a repair. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPEquiv
open VeriTile.Triton FP.Structural FPValue Host HostSchedule
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete-source memory equivalence, including execution success and frame;
this is derived from the independent initial-byte target for both executions. -/
specification fp4_to_bf16_schedule_fp_equivalence
    (left right : Algebra Word) (h : Input) (a b : List Nat) (initial : State Word)
    (fresh : h.input ≠ h.output) (typed : PackedInput h initial)
    (ha : a.Perm (List.range (grid h))) (hb : b.Perm (List.range (grid h))) :
    ∃ sa sb, runSchedule left h a initial = some sa ∧
      runSchedule right h b initial = some sb ∧
      sa.mem = sb.mem ∧ sa.mem = expectedMemory h initial ∧
      (∀ r o, (r ≠ h.output ∨ h.shape.count * 2 ≤ o) →
        sa.mem r o = initial.mem r o) := by
  obtain ⟨sa, ra, ma, fa⟩ :=
    ScheduleCorrect.fp4_to_bf16_arbitrary_program_order_correctness left h a initial fresh typed ha
  obtain ⟨sb, rb, mb, _⟩ :=
    ScheduleCorrect.fp4_to_bf16_arbitrary_program_order_correctness right h b initial fresh typed hb
  exact ⟨sa, sb, ra, rb, ma.trans mb.symm, ma, fa⟩

#axiomsClean fp4_to_bf16_schedule_fp_equivalence
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPEquiv
