import bench.tritonbench_g.flash_attn.Proofs.IOValues
import bench.tritonbench_g.flash_attn.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
open VeriTile.Bench.Repairs.AttentionKernel.Whole001.Scatter
noncomputable section
set_option maxHeartbeats 1000000
variable {A B : Type} [Inhabited A] [Inhabited B]

/-- Source local registers are cleared, while arbitrary exterior memory and
undefined-register payloads keep their shared IO meaning. -/
def prepared (x : Layout) (job : Program) (s : State B) : State B :=
  { stateAt (parameters x) job s.mem with undef := s.undef }

theorem invocation_initial (M : Algebra B) (dot : Dot B) (x : Layout) (scale : B)
    (job : Program) (s : State B) :
    (invocation M dot x scale job).initial s=bindScale scale (prepared x job s) := rfl

theorem invocation_memory (M : Algebra B) (dot : Dot B) (x : Layout) (scale : B)
    (job : Program) (s : State B) :
    ∃ t,(invocation M dot x scale job).run s=some t ∧
      t.mem=referenceProgram M dot (parameters x) scale job s.mem := by
  obtain ⟨t,executed,memory⟩ := source_memory M dot (parameters x) (prepared x job s) scale
  refine ⟨t,?_,?_⟩
  · simpa only [Invocation.run,invocation_initial,invocation] using executed
  · rw [memory]
    have acc : accumulated M dot (parameters x) (prepared x job s) scale=
        accumulated M dot (parameters x) (stateAt (parameters x) job s.mem) scale := rfl
    rw [acc]
    simp only [referenceProgram,written,masked_scatter_memory,prepared,active,ActualSource.address,lActive,ActualSource.lAddress,base]

theorem schedule_memory (M : Algebra B) (dot : Dot B) (x : Layout) (scale : B)
    (jobs : List Program) (s : State B) :
    ∃ t,runSchedule (jobs.map (invocation M dot x scale)) s=some t ∧
      t.mem=referenceGrid M dot (parameters x) scale jobs s.mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons job rest ih =>
    obtain ⟨next,he,hm⟩ := invocation_memory M dot x scale job s
    obtain ⟨final,hf,hmf⟩ := ih next
    refine ⟨final,?_,?_⟩
    · simpa only [List.map_cons,runSchedule,he,Option.bind_some] using hf
    · simpa only [referenceGrid,hm] using hmf

theorem schedule_safe (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
    (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)
    (xs : Inputs A x) (bounds : RegionName → Nat)
    (hb : (sourceIO M dot encode x scale observe).InBounds bounds xs)
    (jobs : List Program) (valid : ∀ job∈jobs,job.Valid x) (s : State B) :
    ScheduleSafe bounds (jobs.map (invocation M dot x scale)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    constructor
    · change TraceSafe _ _ _ ((invocation M dot x scale job).initial s)
      rw [invocation_initial]
      apply IOSourceSafety.source_safe
      obtain ⟨q,k,v,o,l⟩ :=
        source_bounds M dot encode x scale observe xs bounds hb job (valid job (by simp)) s.mem
      exact ⟨q,k,v,o,l⟩
    · intro t _
      exact ih (fun j hj => valid j (by simp [hj])) t

theorem binding_preserved (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
    (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)
    (xs : Inputs A x) (s t : State B)
    (hin : (sourceIO M dot encode x scale observe).InputBinding xs s)
    (jobs : List Program) (hm : t.mem=referenceGrid M dot (parameters x) scale jobs s.mem) :
    (sourceIO M dot encode x scale observe).InputBinding xs t := by
  intro port i active
  rw [hm]
  have region : (sourceIO M dot encode x scale observe).inputs[port].buf ∈
      (["Q","K","V"] : List RegionName) := by
    fin_cases port <;> simp [sourceIO,memory,inputs]
  rw [input_frame M dot (parameters x) scale jobs s.mem _ region]
  exact hin port i active

#axiomsClean invocation_memory
#axiomsClean schedule_memory
#axiomsClean schedule_safe
#axiomsClean binding_preserved
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
