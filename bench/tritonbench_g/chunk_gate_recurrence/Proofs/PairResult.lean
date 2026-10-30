import bench.tritonbench_g.chunk_gate_recurrence.PairIO
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.PairSourceSafety

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier PairIO
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Complete executions, both access-safety traces, all active output cells and
both exterior-cell frames. The logical input predicate is True, preserving all
FP32 words and all existing seed strides; safety bounds remain explicit IO data. -/
theorem equivalent (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (nonempty : 0<c.N) :
    (originalIO isa plan kind c s seed p).Equivalent (sumLeftIO isa plan kind c s seed p) := by
  intro vals _ bounds hb initial hin
  obtain ⟨safe,final,executed,output,frame⟩ :=
    PublicIO.implements (model isa plan kind c) kind c s seed p nonempty vals trivial bounds hb initial hin
  have runEq : runSchedule ((originalIO isa plan kind c s seed p).schedule vals) initial =
      runSchedule ((sumLeftIO isa plan kind c s seed p).schedule vals) initial := by
    cases kind <;>
      simpa only [originalIO,IO.program,IO.forward,IO.backward,sumLeftIO,runSchedule_singleton,
        Invocation.run,IO.launch,model] using
        complete_run isa plan (swapped _ c) _ c s seed
          ((IO.launch (model isa plan _ c) _ c s seed p).initial initial)
  have safeRight : ScheduleSafe bounds ((sumLeftIO isa plan kind c s seed p).schedule vals) initial := by
    cases kind with
    | forward =>
      change ScheduleSafe bounds [_] initial at safe ⊢
      rw [scheduleSafe_singleton] at safe ⊢
      exact (complete_safe isa plan (swapped .forward c) .forward c s seed bounds
        ((IO.launch (model isa plan .forward c) .forward c s seed p).initial initial)).mp safe
    | backward =>
      change ScheduleSafe bounds [_] initial at safe ⊢
      rw [scheduleSafe_singleton] at safe ⊢
      exact (complete_safe isa plan (swapped .backward c) .backward c s seed bounds
        ((IO.launch (model isa plan .backward c) .backward c s seed p).initial initial)).mp safe
  refine ⟨safe,safeRight,final,final,executed,runEq.symm.trans executed,?_,frame,frame⟩
  intro o j active
  rfl

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
