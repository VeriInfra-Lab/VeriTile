import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOValues

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {A B : Type} [Inhabited A] [Inhabited B]
variable (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
  (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
  (counts : Fin c.batch → Nat) (work : List Host.Job)

def rawRow (xs : Inputs A c counts) (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) : B :=
  orderedRow (M ⟨b.val,h.val⟩) c.dimensions (counts b)
    (fun k => encode (logitAt xs b h k)) (fun k d => encode (valueAt xs b h k d)) d

theorem source_single (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads)
    (bounds : RegionName → Nat) (hb : (sourceIO M encode c observe counts work).InBounds bounds xs) :
    (invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).Safe bounds s ∧
    ∃ t, (invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).run s=some t ∧
      (∀ d : Fin c.dimensions, t.mem "Out" (Host.address c ⟨b.val,h.val⟩ d.val)=
        .mk .fp32 (rawRow M encode c counts xs b h d)) ∧
      (∀ (r : RegionName) a, (r≠"Out" ∨ ∀ d : Fin c.dimensions, a≠Host.address c ⟨b.val,h.val⟩ d.val) →
        t.mem r a=s.mem r a) := by
  constructor
  · exact IOSourceSafety.source_safe _ _ _ bounds
      (source_bounds M encode c observe counts work xs s bound hm b h bounds hb) (by change 0<64; decide)
  · obtain ⟨t,run,post⟩ := correctness (M ⟨b.val,h.val⟩) (Host.parameters c)
      ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s) (by change 0<64; decide)
      (invocation_injective M c ⟨b.val,h.val⟩ s)
    refine ⟨t,run,?_,?_⟩
    · intro d
      have value := post.1 (d,())
      rw [invocation_offset,row_logical M encode c observe counts work xs s bound hm] at value
      exact value
    · intro r a outside
      have result := post.2 r a (outside.imp_right fun ho d => by
        rw [invocation_offset]; exact ho d.1)
      exact result

def WorkOutside (jobs : List Host.Job) (r : RegionName) (a : Nat) : Prop :=
  r≠"Out" ∨ ∀ j∈jobs, ∀ d : Fin c.dimensions, a≠Host.address c j d.val

theorem grid (xs : Inputs A c counts) (hm : LengthsMatch xs)
    (bounds : RegionName → Nat) (hb : (sourceIO M encode c observe counts work).InBounds bounds xs)
    (jobs : List Host.Job) (legal : ∀ j∈jobs, Host.Legal c j)
    (s : State B) (bound : (sourceIO M encode c observe counts work).InputBinding xs s) :
    ScheduleSafe bounds (jobs.map (fun j => invocation (M j) c j)) s ∧
    ∃ t, runSchedule (jobs.map (fun j => invocation (M j) c j)) s=some t ∧
      (∀ b h, (⟨b.val,h.val⟩ : Host.Job)∈jobs → ∀ d : Fin c.dimensions,
        t.mem "Out" (Host.address c ⟨b.val,h.val⟩ d.val)=
          .mk .fp32 (rawRow M encode c counts xs b h d)) ∧
      (∀ r a, WorkOutside c jobs r a → t.mem r a=s.mem r a) := by
  induction jobs generalizing s with
  | nil => exact ⟨trivial,s,rfl,by simp,fun _ _ _ => rfl⟩
  | cons j rest ih =>
    have hj := legal j (by simp)
    let b : Fin c.batch := ⟨j.batch,hj.1⟩
    let h : Fin c.heads := ⟨j.head,hj.2⟩
    obtain ⟨safe,u,run,values,frame⟩ := source_single M encode c observe counts work xs s bound hm b h bounds hb
    have nextBinding := binding_of_frame M encode c observe counts work xs s u bound
      (fun r a hr => frame r a (Or.inl hr))
    obtain ⟨restSafe,t,tail,outputs,restFrame⟩ := ih (fun k hk => legal k (by simp [hk])) u nextBinding
    refine ⟨?_,t,?_,?_,?_⟩
    · simp only [List.map_cons,ScheduleSafe]
      refine ⟨safe,?_⟩
      intro v hv
      have eq : u=v := Option.some.inj (run.symm.trans hv)
      simpa [←eq] using restSafe
    · have run' : (invocation (M j) c j).run s=some u := run
      simp only [List.map_cons,runSchedule,run',Option.bind_some]
      exact tail
    · intro b' h' member d
      by_cases memTail : (⟨b'.val,h'.val⟩ : Host.Job)∈rest
      · exact outputs b' h' memTail d
      · have eq : (⟨b'.val,h'.val⟩ : Host.Job)=j := by simpa [memTail] using member
        have saved := restFrame "Out" (Host.address c ⟨b'.val,h'.val⟩ d.val) (Or.inr (by
          intro k hk e same
          have own := Host.ownership c ⟨b'.val,h'.val⟩ k ⟨b'.isLt,h'.isLt⟩
            (legal k (by simp [hk])) d.val e.val d.isLt e.isLt same
          exact memTail (own.1.symm ▸ hk)))
        rw [saved]
        have eb : b'=b := Fin.ext (congrArg Host.Job.batch eq)
        have eh : h'=h := Fin.ext (congrArg Host.Job.head eq)
        simpa [eb,eh] using values d
    · intro r a outside
      have last := restFrame r a (outside.imp_right fun ho k hk => ho k (by simp [hk]))
      have first := frame r a (outside.imp_right fun ho => ho j (by simp))
      exact last.trans first

theorem ordered_implements (complete : work.Perm (Host.workset c)) :
    (sourceIO M encode c observe counts work).ImplementsOn LengthsMatch
      (orderedTarget M encode c observe counts) := by
  intro xs hm bounds hb s bound
  obtain ⟨safe,t,run,values,frame⟩ := grid M encode c observe counts work xs hm bounds hb work
    (fun j hj => (Host.mem_workset c j).mp (complete.mem_iff.mp hj)) s bound
  refine ⟨safe,t,run,?_,?_⟩
  · intro port i _
    have port0 : port=(0 : Fin 1) := by
      apply Fin.ext
      change port.val=0
      have hp : port.val<1 := port.isLt
      omega
    subst port
    have member : (⟨i.1.val,i.2.1.val⟩ : Host.Job)∈work :=
      complete.mem_iff.mpr ((Host.mem_workset c _).mpr ⟨i.1.isLt,i.2.1.isLt⟩)
    have value := values i.1 i.2.1 member i.2.2.1
    change mapCell (observe i.1 i.2.1 i.2.2.1)
      (t.mem "Out" (Host.address c ⟨i.1.val,i.2.1.val⟩ i.2.2.1.val))=
      .mk .fp32 (orderedTarget M encode c observe counts xs (0 : Fin 1) i)
    rw [value]
    rfl
  · intro r a outside
    apply frame r a
    by_cases hr : r="Out"
    · right
      intro j hj d same
      have legal := (Host.mem_workset c j).mp (complete.mem_iff.mp hj)
      have off := outside.1 (0 : Fin 1) (⟨j.batch,legal.1⟩,⟨j.head,legal.2⟩,d,()) trivial
      exact off.elim (fun no => no hr) (fun no => no same)
    · exact Or.inl hr

#axiomsClean ordered_implements
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
