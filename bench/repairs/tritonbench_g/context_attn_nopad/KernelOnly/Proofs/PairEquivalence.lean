import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.PairIO
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Proofs.SourcePair
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.PublicFPResult

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.PairEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

private theorem job_run (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s : State α) :
    (IOData.job c r o idx).run s = (PairIO.candidateInvocation c r o idx).run s := by
  exact SourcePair.same_run (FP.ProfiledRewrite.engine o.algebra) c.layout r c.scale _
private theorem job_safe (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (bounds : RegionName → Nat) (s : State α) :
    (IOData.job c r o idx).Safe bounds s ↔ (PairIO.candidateInvocation c r o idx).Safe bounds s := by
  exact SourcePair.same_safe (FP.ProfiledRewrite.engine o.algebra) bounds c.layout r c.scale _

theorem grid_run (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s : State α) :
    runSchedule (order.map (IOData.job c r o)) s =
      runSchedule (order.map (PairIO.candidateInvocation c r o)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih => simp only [List.map_cons,runSchedule,job_run,ih]
theorem grid_safe (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (order.map (IOData.job c r o)) s ↔
      ScheduleSafe bounds (order.map (PairIO.candidateInvocation c r o)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih => simp only [List.map_cons,ScheduleSafe,job_safe,job_run,ih]

/-- Entire selected grid, including incomplete metadata coverage and aliases:
both safe successful executions, all active outputs and both exterior frames. -/
theorem equivalent (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence) :
    PairIO.original c r o ≡[[]] PairIO.blockStartReused c r o := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals metadata bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    IOData.complete c r o startLabel lengthLabel vals metadata bounds bounded initial binding
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (grid_safe c r o (FPHost.jobs c) bounds initial).mp safe
  · exact (grid_run c r o (FPHost.jobs c) initial).symm.trans ran
  · intro port lane _
    rfl
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.PairEquivalence
