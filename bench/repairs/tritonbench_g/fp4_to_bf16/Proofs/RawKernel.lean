import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Launch
import bench.tritonbench_g.fp4_to_bf16.Proofs.Real
namespace VeriTile.Bench.Repairs.Fp4ToBf16.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostSchedule
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Raw complete finite-ABI launch. Storage is only byte capacities, addresses
and disjoint intervals; it contains no Tensor object or API flags. -/
theorem complete (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (initial : State Word) (fresh : h.input≠h.output)
    (typed : PackedInput h initial) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    ∃ final, Launch.run rest h jobs initial=some final ∧
      final.mem=Host.expectedMemory h initial ∧
      (∀ r o,r≠h.output ∨ h.shape.count*2≤o → final.mem r o=initial.mem r o) := by
  rw [Launch.complete_refinement rest h storage jobs initial gridSafe order]
  exact ScheduleCorrect.fp4_to_bf16_arbitrary_program_order_correctness rest h jobs initial fresh typed order

#axiomsClean complete
end
end VeriTile.Bench.Repairs.Fp4ToBf16.RawKernel
