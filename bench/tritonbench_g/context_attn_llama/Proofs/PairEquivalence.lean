import bench.tritonbench_g.context_attn_llama.PairIO
import bench.tritonbench_g.context_attn_llama.Proofs.SourcePair
import bench.tritonbench_g.context_attn_llama.Proofs.NativeImplementation

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.PairEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

private theorem job_run (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid) (s : State α) :
    (NativeInterface.invocation c M idx).run s = (PairIO.candidateInvocation c M idx).run s := by
  exact SourcePair.same_run (M.withDefaultPrecision .fp32) c.layout c.regions (Host.scale c.layout.D) _
private theorem job_safe (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid)
    (bounds : RegionName → Nat) (s : State α) :
    (NativeInterface.invocation c M idx).Safe bounds s ↔
      (PairIO.candidateInvocation c M idx).Safe bounds s := by
  exact SourcePair.same_safe (M.withDefaultPrecision .fp32) bounds c.layout c.regions (Host.scale c.layout.D) _

theorem grid_run (c : Host.Call) (M : Algebra α) (order : List (GridIndex c.grid)) (s : State α) :
    runSchedule (order.map (NativeInterface.invocation c M)) s =
      runSchedule (order.map (PairIO.candidateInvocation c M)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih => simp only [List.map_cons,runSchedule,job_run,ih]
theorem grid_safe (c : Host.Call) (M : Algebra α) (order : List (GridIndex c.grid))
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (order.map (NativeInterface.invocation c M)) s ↔
      ScheduleSafe bounds (order.map (PairIO.candidateInvocation c M)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih => simp only [List.map_cons,ScheduleSafe,job_safe,job_run,ih]

/-- All original paged reads, arbitrary job orders, repetitions and admitted
aliases. Both full executions are safe/successful with every output and frame. -/
theorem equivalent (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid))
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out) :
    PairIO.original c M dots order ≡[[]] PairIO.blockEndFactored c M dots order := by
  apply ArrayProgramIO.Implementation.of_equivalentOn
  intro vals metadata bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    NativeInterface.implements c M dots order separate vals metadata initial binding
  refine ⟨safe bounds bounded,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (grid_safe c M order bounds initial).mp (safe bounds bounded)
  · exact (grid_run c M order initial).symm.trans ran
  · intro port lane _
    rfl
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.PairEquivalence
