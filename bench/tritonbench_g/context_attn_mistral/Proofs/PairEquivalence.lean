import bench.tritonbench_g.context_attn_mistral.PairIO
import bench.tritonbench_g.context_attn_mistral.Proofs.SourcePair
import bench.tritonbench_g.context_attn_mistral.Proofs.NativeImplementation

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.PairEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

private theorem job_run (c : Host.Call) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s : State α) :
    (NativeInterface.invocation c o idx).run s = (PairIO.candidateInvocation c o idx).run s := by
  exact SourcePair.same_run (FP.ProfiledRewrite.engine o.algebra) c.layout c.regions (Host.scale c.layout.D) _

private theorem job_safe (c : Host.Call) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (bounds : RegionName → Nat) (s : State α) :
    (NativeInterface.invocation c o idx).Safe bounds s ↔
      (PairIO.candidateInvocation c o idx).Safe bounds s := by
  exact SourcePair.same_safe (FP.ProfiledRewrite.engine o.algebra) bounds
    c.layout c.regions (Host.scale c.layout.D) _

/-- Arbitrary order, repetitions and aliases use the actual preceding state. -/
theorem grid_run (c : Host.Call) (o : FPBackend.Ops α)
    (jobs : List (GridIndex c.grid)) (s : State α) :
    runSchedule (jobs.map (NativeInterface.invocation c o)) s =
      runSchedule (jobs.map (PairIO.candidateInvocation c o)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih => simp only [List.map_cons,runSchedule,job_run,ih]

theorem grid_safe (c : Host.Call) (o : FPBackend.Ops α)
    (jobs : List (GridIndex c.grid)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (NativeInterface.invocation c o)) s ↔
      ScheduleSafe bounds (jobs.map (PairIO.candidateInvocation c o)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih => simp only [List.map_cons,ScheduleSafe,job_safe,job_run,ih]

/-- Complete source schedules: successful safe runs, every typed output cell
and both exterior frames, on the unchanged all-expression array input domain. -/
theorem equivalent (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence) :
    PairIO.original c o jobs ≡[[]] PairIO.blockStartReused c o jobs := by
  apply ArrayProgramIO.Implementation.of_equivalentOn
  intro vals metadata bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    NativeInterface.implements c o jobs separate vals metadata initial binding
  refine ⟨safe bounds bounded,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (grid_safe c o jobs bounds initial).mp (safe bounds bounded)
  · exact (grid_run c o jobs initial).symm.trans ran
  · intro port lane _
    rfl

#axiomsClean grid_run
#axiomsClean grid_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.PairEquivalence
