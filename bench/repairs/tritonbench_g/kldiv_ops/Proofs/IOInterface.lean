import bench.repairs.tritonbench_g.kldiv_ops.Proofs.IOSchedule

namespace VeriTile.Bench.Repairs.KldivOps.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α β : Type} [Inhabited α] [Inhabited β]

theorem forward_implements (M : Algebra β) (p : Parameters) (flag : Bool) (red : Nat)
    (eps : β) (positive : 0<p.block) (encode : α → β) (observe : β → α)
    (f : ForwardValues α p → Fin (outputLength p red) → α)
    (computed : ∀ (values : ForwardValues α p) (initial : State β),
      (∀ i,i<p.rows*p.cols → (initial.mem p.Y i).read .real=encode (predictionAt values i)) →
      (∀ i,i<p.rows*p.cols → (initial.mem p.T i).read .real=encode (targetAt values i)) →
      ∃ final,Program.grid M p flag red eps initial=some final ∧
        (∀ i : Fin (outputLength p red),mapCell observe (final.mem p.O i.val)=Cell.mk .real (f values i)) ∧
        (∀ (r : RegionName) i,r≠p.O ∨ outputLength p red ≤ i → final.mem r i=initial.mem r i)) :
    (forwardIO M p flag red eps encode observe).Implements (fun values => result (f values)) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply forward_schedule_safe M p flag red eps positive (List.range p.rows)
      (fun row member => List.mem_range.mp member) bounds
    · intro i hi
      exact hb.2.1 ⟨0,by simp [forwardIO,forwardMemory,forwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.1 ⟨1,by simp [forwardIO,forwardMemory,forwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 ⟨0,by simp [forwardIO,forwardMemory,outputChannels]⟩ (⟨i,hi⟩,PUnit.unit) trivial
  · intro vals initial hin
    have hy (i : Nat) (hi : i<p.rows*p.cols) :
        (initial.mem p.Y i).read .real=encode (predictionAt vals i) := by
      simpa only [forwardIO,forwardMemory,forwardInputs,ProgramIO.InputBinding,mapValue,
        forwardIndex,predictionAt,dif_pos hi,prediction] using
        hin ⟨0,by simp [forwardIO,forwardMemory,forwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    have ht (i : Nat) (hi : i<p.rows*p.cols) :
        (initial.mem p.T i).read .real=encode (targetAt vals i) := by
      simpa only [forwardIO,forwardMemory,forwardInputs,ProgramIO.InputBinding,mapValue,
        forwardIndex,targetAt,dif_pos hi,target] using
        hin ⟨1,by simp [forwardIO,forwardMemory,forwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,frame⟩ := computed vals initial hy ht
    refine ⟨final,?_,?_,?_⟩
    · exact (forward_schedule_run M p flag red eps (List.range p.rows) initial).trans ran
    · intro o j _
      have ho : o.val<1 := o.isLt
      rcases o with ⟨o,ho⟩
      have same : o=0 := by simpa [outputChannels] using Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ ho)
      subst o
      exact values j.1
    · intro r i outside
      apply frame r i
      by_cases different : r≠p.O
      · exact .inl different
      · have same : r=p.O := Classical.not_not.mp different
        subst r
        right
        by_contra h
        have bound : i<outputLength p red := Nat.lt_of_not_ge h
        have no := outside.1 ⟨0,by simp [forwardIO,forwardMemory,outputChannels]⟩ (⟨i,bound⟩,PUnit.unit) trivial
        exact no.elim (fun bad => bad rfl) (fun bad => bad rfl)

theorem backward_implements (M : Algebra β) (p : Parameters) (flag : Bool)
    (positive : 0<p.block) (encode : α → β) (observe : β → α)
    (f : BackwardValues α p → Fin (p.rows*p.cols) → α)
    (computed : ∀ (values : BackwardValues α p) (initial : State β),
      (∀ i,i<p.rows*p.cols → (initial.mem p.T i).read .real=encode (backwardTargetAt values i)) →
      ∃ final,Backward.grid M p flag initial=some final ∧
        (∀ i : Fin (p.rows*p.cols),mapCell observe (final.mem p.O i.val)=Cell.mk .real (f values i)) ∧
        (∀ (r : RegionName) i,r≠p.O ∨ p.rows*p.cols ≤ i → final.mem r i=initial.mem r i)) :
    (backwardIO M p flag encode observe).Implements (fun values => result (f values)) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply backward_schedule_safe M p flag positive (List.range p.rows)
      (fun row member => List.mem_range.mp member) bounds
    · intro i hi
      exact hb.2.1 ⟨0,by simp [backwardIO,backwardMemory,backwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 ⟨0,by simp [backwardIO,backwardMemory,outputChannels]⟩ (⟨i,hi⟩,PUnit.unit) trivial
  · intro vals initial hin
    have ht (i : Nat) (hi : i<p.rows*p.cols) :
        (initial.mem p.T i).read .real=encode (backwardTargetAt vals i) := by
      simpa only [backwardIO,backwardMemory,backwardInputs,ProgramIO.InputBinding,mapValue,
        backwardIndex,backwardTargetAt,dif_pos hi,backwardTarget] using
        hin ⟨0,by simp [backwardIO,backwardMemory,backwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,frame⟩ := computed vals initial ht
    refine ⟨final,?_,?_,?_⟩
    · exact (backward_schedule_run M p flag (List.range p.rows) initial).trans ran
    · intro o j _
      have ho : o.val<1 := o.isLt
      rcases o with ⟨o,ho⟩
      have same : o=0 := by simpa [outputChannels] using Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ ho)
      subst o
      exact values j.1
    · intro r i outside
      apply frame r i
      by_cases different : r≠p.O
      · exact .inl different
      · have same : r=p.O := Classical.not_not.mp different
        subst r
        right
        by_contra h
        have bound : i<p.rows*p.cols := Nat.lt_of_not_ge h
        have no := outside.1 ⟨0,by simp [backwardIO,backwardMemory,outputChannels]⟩ (⟨i,bound⟩,PUnit.unit) trivial
        exact no.elim (fun bad => bad rfl) (fun bad => bad rfl)
end
end VeriTile.Bench.Repairs.KldivOps.Interface
