import bench.tritonbench_g.token_attn_mistral.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.PublicIO
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The new schedule has exactly the old complete grid's memory result. -/
theorem execute (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat) (vals : Ports.Values α p work counts)
    (s : State α) (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    ∃ t,runSchedule ((IO.program M R p work counts).schedule vals) s=some t ∧
      (∀ j : Fin work.length,∀ d : Fin p.width,
        t.mem p.output (Grid.offset p work[j] (d,()))=
          Cell.mk .fp32 (outputValues M R p (Grid.state s.mem work[j]) (d,()))) ∧
      (IO.program M R p work counts).Frame vals s t := by
  obtain ⟨t,ran,mem⟩ := IOExecution.complete M R p work s
  obtain ⟨final,grid,_,values,frame⟩ := Grid.whole M R p work s.mem separate unique
  rw [Grid.complete] at grid
  injection grid with eq
  have finalMem : t.mem=final := mem.trans eq
  refine ⟨t,ran,?_,?_⟩
  · intro j d
    rw [finalMem]
    exact values (work[j],d,()) ((Grid.mem_events p work _).mpr (List.getElem_mem _))
  · intro r o outside
    rw [finalMem]
    apply frame
    by_cases same : r=p.output
    · right
      intro e he
      obtain ⟨j,hj⟩ := List.mem_iff_get.mp ((Grid.mem_events p work e).mp he)
      change work[j]=e.1 at hj
      have h := outside.1 ⟨0,by simp [IO.program,Ports.memory,Ports.outputs]⟩ (j,e.2.1,()) (by change True; trivial)
      have h' : r≠p.output ∨ o≠Grid.offset p work[j] (e.2.1,()) := by
        simpa [IO.program,Ports.memory,Ports.outputs] using h
      simpa [same,hj] using h'
    · exact Or.inl same

theorem implements (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    (IO.program M R p work counts).ImplementsOn (Ports.Domain p work counts)
      (Reference.formula M R p work counts) := by
  intro vals domain bounds hb s binding
  have f := Binding.facts M R p work counts vals domain s binding
  refine ⟨IOSafety.schedule M R p work counts vals bounds hb separate work (fun _ h=>h) s f,?_⟩
  obtain ⟨t,ran,values,frame⟩ := execute M R p work counts vals s separate unique
  refine ⟨t,ran,?_,frame⟩
  intro o i active
  have ho : o=⟨0,by simp [IO.program,Ports.memory,Ports.outputs]⟩ := by
    apply Fin.ext
    change o.val=0
    have h : o.val<1 := o.isLt
    omega
  subst o
  rcases i with ⟨j,d,⟨⟩⟩
  change mapCell id (t.mem p.output (Grid.offset p work[j] (d,())))=
    Cell.mk .fp32 (Reference.result M R p (Ports.logical p work counts vals) j d)
  rw [values, f.result_eq M R domain]
  rfl

theorem real (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (positive : 0<p.block) (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    (IO.real p work counts).ImplementsOn (Ports.Domain p work counts)
      (Reference.mathematical p work counts) := by
  intro vals domain bounds hb s binding
  have f := Binding.facts Real.algebra Real.sum p work counts vals domain s binding
  obtain ⟨safe,t,ran,output,frame⟩ := implements Real.algebra Real.sum p work counts separate unique vals domain bounds hb s binding
  refine ⟨safe,t,ran,?_,frame⟩
  intro o i active
  have ho : o=⟨0,by simp [IO.real,IO.program,Ports.memory,Ports.outputs]⟩ := by
    apply Fin.ext
    change o.val=0
    have h : o.val<1 := o.isLt
    omega
  subst o
  rcases i with ⟨j,d,⟨⟩⟩
  have out := output ⟨0,by simp [IO.program,Ports.memory,Ports.outputs]⟩ (j,d,()) (by change True; trivial)
  change mapCell id (t.mem p.output (Grid.offset p work[j] (d,())))=
    Cell.mk .fp32 (Reference.weighted p (Ports.logical p work counts vals) j d)
  change mapCell id (t.mem p.output (Grid.offset p work[j] (d,())))=
    Cell.mk .fp32 (Reference.result Real.algebra Real.sum p (Ports.logical p work counts vals) j d) at out
  rw [← f.result_eq Real.algebra Real.sum domain,Real.output_weighted p _ positive,
    Binding.real_weighted f domain positive] at out
  exact out


#axiomsClean real
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.PublicIO
