import bench.tritonbench_g.context_attn_fwd.PairIO
import bench.tritonbench_g.context_attn_fwd.Proofs.SourcePair
import bench.tritonbench_g.context_attn_fwd.Proofs.NativeIOComplete

namespace VeriTile.Bench.Repairs.ContextAttnFwd.PairEquivalence
open VeriTile.Triton FP.Structural NativeFormula
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

private theorem job_run (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (job : Launch) (s : State α) :
    (NativeArrayBridge.invocation p r scale M job).run s =
      (PairIO.candidateInvocation p r scale M job).run s := by
  exact SourcePair.same_run (M.withDefaultPrecision .fp32) p r scale _
private theorem job_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (job : Launch) (bounds : RegionName → Nat) (s : State α) :
    (NativeArrayBridge.invocation p r scale M job).Safe bounds s ↔
      (PairIO.candidateInvocation p r scale M job).Safe bounds s := by
  exact SourcePair.same_safe (M.withDefaultPrecision .fp32) bounds p r scale _

theorem grid_run (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (jobs : List Launch) (s : State α) :
    runSchedule (jobs.map (NativeArrayBridge.invocation p r scale M)) s =
      runSchedule (jobs.map (PairIO.candidateInvocation p r scale M)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih => simp only [List.map_cons,runSchedule,job_run,ih]
theorem grid_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (jobs : List Launch) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (NativeArrayBridge.invocation p r scale M)) s ↔
      ScheduleSafe bounds (jobs.map (PairIO.candidateInvocation p r scale M)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih => simp only [List.map_cons,ScheduleSafe,job_safe,job_run,ih]

/-- Every actual arbitrary-job run on the retained raw-array domain: both safe
successful executions, every typed final output and both complete frames. -/
theorem equivalent (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (positive : 0<p.BN) :
    PairIO.original p r M dots scale jobs ≡[[]] PairIO.blockEndFactored p r M dots scale jobs := by
  apply ArrayProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    NativeIOComplete.complete p r M dots scale jobs positive vals trivial initial binding
  refine ⟨safe bounds bounded,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (grid_safe p r scale M jobs bounds initial).mp (safe bounds bounded)
  · exact (grid_run p r scale M jobs initial).symm.trans ran
  · intro port lane _
    rfl
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ContextAttnFwd.PairEquivalence
