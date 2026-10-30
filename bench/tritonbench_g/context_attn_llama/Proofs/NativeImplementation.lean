import bench.tritonbench_g.context_attn_llama.Proofs.NativeSchedule
import bench.tritonbench_g.context_attn_llama.Proofs.NativeOutputTags

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural NativeBindings
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem bound_snapshot (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (values : NativeArrays.Inputs c α) (s : State α)
    (bound : (programIO c M dots order).InputBinding values s) : snapshot c s=values := by
  have identity (dtype : TileDType) (x : Value α dtype) : mapValue id dtype x=x := by
    cases dtype <;> rfl
  funext port off
  have h := bound port off
  change (s.mem (NativeArrays.ports c)[port].buf off).read (NativeArrays.ports c)[port].dtype=
    mapValue id (NativeArrays.ports c)[port].dtype (values port off) at h
  rw [identity] at h
  exact h

theorem implements (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid))
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out) :
    (programIO c M dots order).ImplementsOn (NativeArrays.metadata c) (formula c M dots order) := by
  intro values metadata initial bound
  have snap := bound_snapshot c M dots order values initial bound
  have initialMetadata : FPSuccess.Metadata c initial := by
    rw [←metadata_iff, snap]
    exact metadata
  refine ⟨?_,?_⟩
  · intro bounds backed
    apply safe c M dots order initial initialMetadata separate bounds
    · intro port off member
      rw [snap] at member
      exact backed.2.1 port off member
    · intro idx member i active
      obtain ⟨job,rfl⟩ := List.mem_iff_get.mp member
      exact backed.2.2.1 (0 : Fin 1) (job,i) active
  · obtain ⟨final,ran,memory⟩ := FPReference.whole c M dots order initial
    refine ⟨final,?_,?_,?_⟩
    · exact (run_eq c M order initial).trans ran
    · intro port j active
      fin_cases port
      change mapCell id (final.mem c.regions.out (c.offset order[j.1] j.2))=
        Cell.mk .real (NativeArrays.grid c M dots order values (8 : Fin 9) (c.offset order[j.1] j.2))
      rw [memory]
      have written : FPSuccess.writes c order c.regions.out (c.offset order[j.1] j.2) :=
        ⟨order[j.1],List.getElem_mem j.1.isLt,j.2,active,rfl,rfl⟩
      obtain ⟨x,tag⟩ := grid_real c M dots order initial initialMetadata separate _ written
      have result := congrFun (congrFun (grid_snapshot c M dots order initial initialMetadata separate)
        (8 : Fin 9)) (c.offset order[j.1] j.2)
      rw [snap] at result
      change ((FPReference.grid c M dots order initial).mem c.regions.out _).read .real=_ at result
      rw [tag,Cell.read_same] at result
      rw [tag]
      exact congrArg (Cell.mk .real) result
    · intro region off outside
      rw [memory]
      apply RawKernel.reference_frame c M dots order initial initialMetadata separate
      rintro ⟨idx,member,i,active,same,address⟩
      obtain ⟨job,rfl⟩ := List.mem_iff_get.mp member
      have h := outside.1 (0 : Fin 1) (job,i) active
      exact h.elim (fun h=>h same) (fun h=>h address)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
