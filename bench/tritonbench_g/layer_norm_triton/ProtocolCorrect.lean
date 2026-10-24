import bench.tritonbench_g.layer_norm_triton.Proofs.Protocol

namespace VeriTile.Bench.TritonBenchG.LayerNormLock.ProtocolCorrect
open VeriTile.Triton Frontend
open FP.Equational (ReductionPlan)

/-- The complete generated source, starting from the host's zero lock/count
allocation, preserves SC mutual exclusion for every finite schedule. This is a
proof of the chosen SC profile, not PTX memory-model refinement or termination. -/
specification compiled_source_mutual_exclusion {V : Type} (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (inputs : Memory V) (undef : Undef V)
    (mem : Shared V) (schedule : List Nat) :
    ∃ events finalState,
      compileBackward originalTypedSource b c plan inputs undef mem (fun _ => 0) schedule =
        some (events, finalState) ∧ ProtocolInvariant c finalState.control := by
  let rows := backwardSource b c plan inputs undef
  let result := run b c rows undef schedule (initial rows mem (fun _ => 0))
  have he : compileBackward originalTypedSource b c plan inputs undef mem (fun _ => 0) schedule = some result :=
    original_backward_execution b c plan inputs undef mem (fun _ => 0) schedule
  refine ⟨result.1, result.2, ?_, ?_⟩
  · simpa only [Prod.mk.eta] using he
  · have hc : result.2.control = (runControl c schedule (initialControl (fun _ => 0))).2 :=
      (congrArg Prod.snd (run_control_projection b c rows undef schedule (initial rows mem (fun _ => 0)))).symm
    rw [hc]
    exact zero_initialization_mutual_exclusion c schedule

#axiomsClean compiled_source_mutual_exclusion
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.ProtocolCorrect
