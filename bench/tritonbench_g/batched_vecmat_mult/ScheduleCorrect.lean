import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostSchedule

/-! Exact original fp32 source execution for every permutation of the complete
logical grid. The output region must be logically distinct from A and B;
the input regions may alias each other. This condition is explicit rather
than inferred from an unverified allocator or physical pointer separation.
Only complete sequential program order is covered. Native byte memory,
allocator freshness, overflow, compiler admission and instruction/warp
interleavings remain separate structural obligations. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ScheduleCorrect
open VeriTile.Triton FP.Structural ValueProofs
open FP.Equational (ReductionPlan)
noncomputable section
local instance scheduleCorrectWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Each actual program reads prior actual memory with fresh local registers.
The result is the independent initial-input reference and the previously
published complete ordered-grid reference, with full untouched-memory frame.
The same FP configuration and valid reduction plans are retained; no result,
successful-run or floating reassociation law is assumed. -/
specification batched_vecmat_arbitrary_program_order_correctness
    (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (jobs : List Nat) (initial : State Word)
    (separate : InputsSeparate p) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, runSchedule rest cfg plans p initial jobs = some final ∧
      final.mem = completeInitialMemory cfg plans p initial ∧
      final.mem = gridExpectedMemory cfg plans p initial ∧
      (∀ r o, GridOutside p r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hm, hf⟩ := complete_schedule_correct rest cfg plans p initial separate jobs order
  exact ⟨final, he, hm, hm.trans (complete_initial_memory_eq_grid cfg plans p initial separate), hf⟩

#axiomsClean batched_vecmat_arbitrary_program_order_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ScheduleCorrect
