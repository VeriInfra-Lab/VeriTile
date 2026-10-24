import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardGrid
import bench.repairs.tritonbench_g.cross_entropy2.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropy2.BackwardGrid
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.Grid (Address Task flat unflat tasks task_mem task_nodup flat_unflat unflat_flat)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

abbrev column (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters
abbrev logitAddress (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters
abbrev gradientAddress (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Source.gradientAddress p.toParameters
abbrev label {α : Type} [Inhabited α] (p : Parameters) := @VeriTile.Bench.Repairs.CrossEntropy1.Source.label α _ p.toParameters
abbrev loadedLogits {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) :=
  VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled M p.logitScale) p.toParameters
abbrev backwardReads (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Grid.backwardReads p.toParameters
abbrev backwardWrites (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Grid.backwardWrites p.toParameters

/-- The source has the same single masked scatter, with the new scaled values.
The general scatter frame proof is reused unchanged. -/
theorem program_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (r : RegionName) (offset : Nat)
    (outside : ∀ i : Fin p.block, column p block i.val<p.cols →
      r≠p.gradient ∨ offset≠gradientAddress p row (column p block i.val)) :
    Source.backward M p memory row block r offset=memory r offset := by
  apply VeriTile.Bench.Repairs.CrossEntropy1.Program.scatter_frame
  intro i _ ha
  exact outside i ha

instance (p : Parameters) (q : Task) : DecidablePred (backwardWrites p q) := Classical.decPred _

def backwardValue {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (a : Address) : Source.Cell α :=
  if h : backwardWrites p q a then .mk .real (backwardValues M p entry q.1 q.2 (Classical.choose h))
  else .mk .real (fp M 0)

def backwardStep {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Source.Cell α) : Address → Source.Cell α :=
  flat (Source.backward M p (unflat memory) q.1 q.2)

theorem backward_value_lane {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (lane : Fin p.block) (active : column p q.2 lane.val<p.cols) :
    backwardValue M p entry q (p.gradient,gradientAddress p q.1 (column p q.2 lane.val))=
      .mk .real (backwardValues M p entry q.1 q.2 lane) := by
  have present : backwardWrites p q (p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) :=
    ⟨lane,active,rfl⟩
  rw [backwardValue,dif_pos present]
  have selected := (Classical.choose_spec present).2
  have index := congrArg Prod.snd selected
  have same : Classical.choose present=lane := by
    apply Fin.ext
    simpa only [gradientAddress,column,VeriTile.Bench.Repairs.CrossEntropy1.Source.gradientAddress,VeriTile.Bench.Repairs.CrossEntropy1.Source.column,Nat.add_left_cancel_iff] using index.symm
  rw [same]

theorem backward_cell {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (lane : Fin p.block) (active : column p q.2 lane.val<p.cols) :
    Source.backward M p entry q.1 q.2 p.gradient (gradientAddress p q.1 (column p q.2 lane.val))=
      .mk .real (backwardValues M p entry q.1 q.2 lane) := by
  have unique : Function.Injective (fun i : Fin p.block => gradientAddress p q.1 (column p q.2 i.val)) := by
    intro i j h
    apply Fin.ext
    dsimp [gradientAddress,column,VeriTile.Bench.Repairs.CrossEntropy1.Source.gradientAddress,VeriTile.Bench.Repairs.CrossEntropy1.Source.column] at h
    omega
  have result := VeriTile.Bench.Repairs.CrossEntropy1.Program.scatter_read p.gradient
    (fun i : Fin p.block => gradientAddress p q.1 (column p q.2 i.val))
    (fun i => VeriTile.Bench.Repairs.CrossEntropy1.Source.Cell.mk .real (backwardValues M p entry q.1 q.2 i))
    (fun i : Fin p.block => column p q.2 i.val<p.cols) unique
    (List.ofFn (fun i : Fin p.block => i)) entry lane active
  simpa only [List.mem_ofFn,exists_apply_eq_apply,if_true] using result

theorem backward_form {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) :
    backwardStep M p q (flat entry)=
      VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put (backwardWrites p) (backwardValue M p entry) q (flat entry) := by
  funext a
  by_cases active : backwardWrites p q a
  · obtain ⟨lane,ha,rfl⟩ := active
    have active : backwardWrites p q (p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) := ⟨lane,ha,rfl⟩
    simp only [VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put,if_pos active,backward_value_lane M p entry q lane ha]
    exact backward_cell M p entry q lane ha
  · simp only [VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put,if_neg active]
    apply program_frame
    intro lane ha
    have miss : a≠(p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) := by
      intro h
      exact active ⟨lane,ha,h⟩
    by_contra h
    push Not at h
    exact miss (Prod.ext h.1 h.2)

theorem backward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Source.Cell α) (a : Address) (outside : ¬backwardWrites p q a) :
    backwardStep M p q memory a=memory a := by
  have form := congrFun (backward_form M p (unflat memory) q) a
  simpa only [flat_unflat,VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put,if_neg outside] using form

theorem backward_value_stable {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry current : Memory α) (q : Task)
    (same : ∀ a, backwardReads p q a → flat current a=flat entry a) :
    backwardValue M p current q=backwardValue M p entry q := by
  have labelsSame := same (Region.cast p.labels,p.labelsBase+q.1) (Or.inl rfl)
  have upstreamSame := same (p.upstream,p.upstreamBase+q.1*p.upstreamStride) (Or.inr (Or.inl rfl))
  have lseSame := same (p.lse,p.lseBase+q.1) (Or.inr (Or.inr (Or.inl rfl)))
  have loadedEq : loadedLogits M p current q.1 q.2=loadedLogits M p entry q.1 q.2 := by
    funext lane
    by_cases ha : column p q.2 lane.val<p.cols
    · have he := same (p.logits,logitAddress p q.1 (column p q.2 lane.val))
        (Or.inr (Or.inr (Or.inr ⟨lane,ha,rfl⟩)))
      simp only [flat] at he
      simp [loadedLogits,VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits,ha,Source.load,VeriTile.Bench.Repairs.CrossEntropy1.Source.load,he]
    · simp [loadedLogits,VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits,ha]
  simp only [flat] at labelsSame upstreamSame lseSame
  have valuesEq : backwardValues M p current q.1 q.2=backwardValues M p entry q.1 q.2 := by
    funext lane
    simp [Source.backwardValues,loadedLogits,loadedEq,label,VeriTile.Bench.Repairs.CrossEntropy1.Source.label,Source.load,VeriTile.Bench.Repairs.CrossEntropy1.Source.load,labelsSame,upstreamSame,lseSame]
  funext a
  simp only [backwardValue,valuesEq]

/-- Exact geometry and inter-program separation from the frozen helper. -/
abbrev BackwardSeparation (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Grid.BackwardSeparation p.toParameters
abbrev gradient_unique (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Grid.gradient_unique p.toParameters
abbrev backward_read_separation (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Grid.backward_read_separation p.toParameters

theorem backward_frozen {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : BackwardSeparation p) :
    VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (backwardStep M p) (tasks p.rows count) (flat entry)=
      VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put (backwardWrites p) (backwardValue M p entry))
        (tasks p.rows count) (flat entry) := by
  apply VeriTile.Bench.Repairs.CrossEntropy1.Schedule.freeze_reads (reads:=backwardReads p) (writes:=backwardWrites p) (entry:=flat entry)
  · intro q memory same
    have form := backward_form M p (unflat memory) q
    have values := backward_value_stable M p entry (unflat memory) q same
    funext a
    simpa only [flat_unflat,VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put,values] using congrFun form a
  · exact backward_frame M p
  · exact task_nodup _ _
  · exact backward_read_separation p count separation
  · intro _ _ _ _
    rfl

def backward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) : Memory α :=
  unflat (VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (backwardStep M p) (tasks p.rows count) (flat entry))

theorem backward_output {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (q : Task) (present : q∈tasks p.rows count)
    (separation : BackwardSeparation p) (lane : Fin p.block) (active : column p q.2 lane.val<p.cols) :
    backward M p entry count p.gradient (gradientAddress p q.1 (column p q.2 lane.val))=
      .mk .real (backwardValues M p entry q.1 q.2 lane) := by
  have unique : ∀ other∈tasks p.rows count,
      backwardWrites p other (p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) → other=q := by
    intro other ho hw
    obtain ⟨otherLane,hactive,haddr⟩ := hw
    have same := congrArg Prod.snd haddr
    exact (gradient_unique p other q otherLane lane separation.rows
      ((task_mem _ _ _).mp ho).1 ((task_mem _ _ _).mp present).1 hactive active same.symm).1
  have output := VeriTile.Bench.Repairs.CrossEntropy1.Schedule.fixed_output (backwardWrites p) (backwardValue M p entry) (tasks p.rows count)
    (flat entry) q (p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) ⟨lane,active,rfl⟩ unique
  simp only [backward,unflat,backward_frozen M p entry count separation]
  simpa only [present,if_true,backward_value_lane M p entry q lane active] using output

theorem backward_complete_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : BackwardSeparation p) (r : RegionName) (i : Nat)
    (outside : ∀ q∈tasks p.rows count, ¬backwardWrites p q (r,i)) :
    backward M p entry count r i=entry r i := by
  simp only [backward,unflat,backward_frozen M p entry count separation]
  exact VeriTile.Bench.Repairs.CrossEntropy1.Schedule.fixed_frame _ _ _ _ _ outside

abbrev owner (p : Parameters) := VeriTile.Bench.Repairs.CrossEntropy1.Grid.owner p.toParameters

#axiomsClean backward_output
#axiomsClean backward_complete_frame
#axiomsClean backward_frozen
#axiomsClean gradient_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.BackwardGrid
