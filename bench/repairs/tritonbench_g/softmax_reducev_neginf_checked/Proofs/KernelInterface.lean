import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Completion

/-! Raw complete-grid execution helpers, shared by the main kernel statements
and the Tensor application adapter. -/
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
open VeriTile.Triton FP.Structural
open SoftmaxTriton1 (Result)
noncomputable section
/-- A local call executes its complete source evaluator. There is no numerical
or final-output callback: these calls are instantiated below by concrete ASTs. -/
def sourceCall {α : Type} (evaluate : Host.Job → (Nat → α) → Option (Nat → α))
    (j : Host.Job) (s : Nat → α) : Result (Nat → α) Unit :=
  match evaluate j s with
  | some t => .ok () t
  | none => .error "source execution" s

theorem source_runtime {α : Type} (c : Host.Call)
    (evaluate : Host.Job → (Nat → α) → Option (Nat → α)) :
    Host.RuntimeGeneral c evaluate (sourceCall evaluate) := by
  constructor
  · intro j s t legal run
    cases eq : evaluate j s with
    | none => simp [sourceCall,eq] at run
    | some u =>
      have same : u=t := by simpa [sourceCall,eq] using run
      simpa [eq,same]
  · intro j s e t legal run
    cases eq : evaluate j s with
    | some u => simp [sourceCall,eq] at run
    | none =>
      have both : "source execution"=e ∧ s=t := by simpa [sourceCall,eq] using run
      have same := both.2
      subst t
      exact fun _ _ => rfl

theorem source_jobs_total {α : Type} (evaluate : Host.Job → (Nat → α) → Option (Nat → α))
    (total : ∀ j s,∃ t,evaluate j s=some t) (work : List Host.Job) (s : Nat → α) :
    ∃ t,Host.runJobs (sourceCall evaluate) work s=.ok () t := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons j rest ih =>
    obtain ⟨u,run⟩ := total j s
    obtain ⟨t,tail⟩ := ih u
    exact ⟨t,by simp [Host.runJobs,sourceCall,run,Result.bind,tail]⟩

theorem source_whole {α : Type} [Inhabited α] (c : Host.Call)
    (evaluate : Host.Job → (Nat → α) → Option (Nat → α)) (target : Host.Job → Nat → α)
    (source : Host.SourceLawGeneral c evaluate target) (total : ∀ j s,∃ t,evaluate j s=some t)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) (m : Host.Memory α) :
    ∃ final,Host.wrapper c 0 work (sourceCall evaluate) m=.ok () final ∧
      (∀ j,Host.Legal c j → ∀ d,d < c.dimensions → final "Out" (Host.address c j d)=target j d) ∧
      (∀ r i,Host.Outside c 0 r i → final r i=m r i) := by
  by_cases empty : Host.noWork c
  · refine ⟨m,by simp [Host.wrapper,empty],?_,fun _ _ _ => rfl⟩
    intro j legal d hd
    rcases empty with a|b <;> dsimp [Host.Legal] at legal <;> omega
  · obtain ⟨out,run⟩ := source_jobs_total evaluate total work (Host.readOutput 0 m)
    have runFull : Host.wrapper c 0 work (sourceCall evaluate) m=.ok () (Host.putOutput 0 m out) := by
      simp [Host.wrapper,empty,Host.runFull_projection,run,Host.liftResult]
    have result := Host.whole_general c evaluate target source 0 (sourceCall evaluate)
      (source_runtime c evaluate) work complete m _ runFull
    exact ⟨_,runFull,by simpa [Host.address,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using result.1,result.2⟩


#axiomsClean source_jobs_total
#axiomsClean source_whole
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.RawKernel
open VeriTile.Triton

/-- Execute every source program in a complete grid order, preserving the raw
caller memory outside its output region. No admission/check wrapper is used. -/
def runGrid {α : Type} (c : Host.Call)
    (evaluate : Host.Job → (Nat → α) → Option (Nat → α))
    (work : List Host.Job) (m : Host.Memory α) :=
  Host.wrapper c 0 work (Checked.sourceCall evaluate) m

end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.RawKernel
