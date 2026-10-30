import bench.tritonbench_g.token_attn_reduceV.Proofs.Bindings

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
variable {A : Type} [Inhabited A]

theorem invocation_reference (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (pid : Grid.PID) (s : State A) :
    (reference M R p ((invocation M R p pid).initial s)).mem =
      (reference M R p (Grid.state s.mem pid)).mem := by
  have hv : outputValues M R p ((invocation M R p pid).initial s)=
      outputValues M R p (Grid.state s.mem pid) := rfl
  have ho : outputOffset p ((invocation M R p pid).initial s)=outputOffset p (Grid.state s.mem pid) := rfl
  rw [reference,fold_write_memory,reference,fold_write_memory,hv,ho]
  rfl

theorem invocation_complete (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (pid : Grid.PID) (s : State A) :
    ∃ t, (invocation M R p pid).run s=some t ∧ t.mem=(reference M R p (Grid.state s.mem pid)).mem := by
  obtain ⟨t,ran,mem⟩ := complete_run M R p ((invocation M R p pid).initial s)
  exact ⟨t,ran,mem.trans (invocation_reference M R p pid s)⟩

theorem invocation_frame (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (separate : Grid.Separate p) (pid : Grid.PID) (s t : State A)
    (ran : (invocation M R p pid).run s=some t) : ∀ r, Grid.Reads p r → t.mem r=s.mem r := by
  obtain ⟨u,execution,mem⟩ := invocation_complete M R p pid s
  have eq : t=u := Option.some.inj (ran.symm.trans execution)
  subst u
  intro r hr
  funext o
  rw [mem]
  exact reference_frame M R p (Grid.state s.mem pid) r o (Or.inl (separate r hr))

theorem schedule_complete (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (jobs : List Grid.PID) (s : State A) :
    ∃ t, runSchedule (jobs.map (invocation M R p)) s=some t ∧ t.mem=Grid.graph M R p jobs s.mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨middle,ran,mem⟩ := invocation_complete M R p pid s
    obtain ⟨t,restRun,restMem⟩ := ih middle
    refine ⟨t,?_,?_⟩
    · simp only [List.map_cons,runSchedule,ran,Option.bind_some,restRun]
    · simpa only [Grid.graph,mem] using restMem

theorem schedule_safe (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (valid : ∀ pid∈jobs, pid.1<B ∧ pid.2<H) (separate : Grid.Separate p)
    (v : Inputs A p B H counts) (domain : LengthsMatch p B H counts v)
    (bounds : RegionName → Nat) (bounded : (programIO M R p B H counts jobs).InBounds bounds v)
    (s : State A) (bound : (programIO M R p B H counts jobs).InputBinding v s) :
    ScheduleSafe bounds (jobs.map (invocation M R p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    have dims := valid pid (by simp)
    let b : Fin B := ⟨pid.1,dims.1⟩
    let h : Fin H := ⟨pid.2,dims.2⟩
    have inputs := call_inputs M R p B H counts (pid::rest) v domain
      ((invocation M R p pid).initial s) bound b h rfl rfl
    refine ⟨SourceSafety.source_safe M R p _ bounds
      (call_bounds M R p B H counts (pid::rest) v bounds bounded _ b h inputs),?_⟩
    intro t ran
    apply ih (fun q hq => valid q (List.mem_cons_of_mem pid hq)) bounded t
    exact bound_after_frame M R p B H counts (pid::rest) v s t bound
      (invocation_frame M R p separate pid s t ran)

#axiomsClean schedule_complete
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
