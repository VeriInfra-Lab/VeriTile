import bench.tritonbench_g.cross_entropy_ops.Proofs.Execution
import bench.tritonbench_g.cross_entropy1.Proofs.BackwardGrid

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Grid
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Memory Cell write)
open CrossEntropy1.Grid (Address Task flat unflat tasks task_mem task_nodup flat_unflat unflat_flat)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

abbrev forwardReads (p : Parameters) := CrossEntropy1.Grid.forwardReads p.toParameters

def forwardWrites (p : Parameters) (q : Task) (a : Address) : Prop :=
  CrossEntropy1.Grid.forwardWrites p.toParameters q a ∨
    (p.split=Bool.false ∧ a.1=p.zloss ∧ a.2=p.zlossBase+q.2*p.rows+q.1)
instance (p : Parameters) (q : Task) : DecidablePred (forwardWrites p q) := by
  intro a
  unfold forwardWrites CrossEntropy1.Grid.forwardWrites
  infer_instance

def forwardValue {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (a : Address) : Cell α :=
  if a.1=p.zloss then .mk .real (Source.zLoss M p
    (CrossEntropy1.Source.label p.toParameters entry q.1) (Source.forwardValues M p entry q.1 q.2).lse)
  else CrossEntropy1.Grid.forwardValue (Source.arithmetic M) p.toParameters
    (Source.scaledReads M p entry Bool.false) q a

def forwardStep {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Cell α) : Address → Cell α :=
  flat (Source.forward M p (unflat memory) q.1 q.2)

structure ForwardSeparation (p : Parameters) : Prop extends CrossEntropy1.Grid.ForwardSeparation p.toParameters where
  zLoss : p.zloss≠p.loss
  zLse : p.zloss≠p.lse
  zLogits : p.zloss≠p.logits
  zLabels : p.zloss≠Region.cast p.labels

/-- Local read substitution preserves equality of complete cells; it adds no
stored tensor, global output premise, or numerical assumption. -/
theorem scaled_cell_congr {α : Type} (M : FP32 α) (p : Parameters)
    (entry current : Memory α) (b : Bool) (a : Address)
    (same : current a.1 a.2=entry a.1 a.2) :
    Source.scaledReads M p current b a.1 a.2=Source.scaledReads M p entry b a.1 a.2 := by
  unfold Source.scaledReads
  rw [same]

theorem forward_form {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (sep : ForwardSeparation p) :
    forwardStep M p q (flat entry)=CrossEntropy1.Schedule.put (forwardWrites p) (forwardValue M p entry) q (flat entry) := by
  have stable : ∀ i, write entry p.lse (p.lseBase+q.2*p.rows+q.1)
      (.mk .real (Source.forwardValues M p entry q.1 q.2).lse) p.logits i=entry p.logits i := by
    intro i
    exact CrossEntropy1.Program.write_other _ _ _ _ _ _ (Or.inl sep.lseLogits.symm)
  have lossEq := Source.forwardLoss_input M p entry _ stable q.1 q.2
    (CrossEntropy1.Source.label p.toParameters entry q.1) (Source.forwardValues M p entry q.1 q.2)
  funext a
  simp only [forwardStep,flat,unflat_flat,Source.forward,lossEq]
  by_cases hz : a.1=p.zloss <;> by_cases hl : a.1=p.loss <;> by_cases he : a.1=p.lse
  all_goals try (exact False.elim (sep.zLoss (hz.symm.trans hl)))
  all_goals try (exact False.elim (sep.zLse (hz.symm.trans he)))
  all_goals try (exact False.elim (sep.outputDistinct (he.symm.trans hl)))
  all_goals cases hs : p.split <;>
    simp [write,CrossEntropy1.Schedule.put,forwardWrites,CrossEntropy1.Grid.forwardWrites,forwardValue,
      CrossEntropy1.Grid.forwardValue,Source.forwardLoss,Source.forwardValues,Source.label_unchanged,
      flat,hz,hl,he,hs,sep.zLoss,sep.zLoss.symm,sep.zLse,sep.zLse.symm,sep.outputDistinct,sep.outputDistinct.symm]

theorem forward_value_stable {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry current : Memory α) (q : Task)
    (same : ∀ a, forwardReads p q a → flat current a=flat entry a) :
    forwardValue M p current q=forwardValue M p entry q := by
  have scaledSame : ∀ a, CrossEntropy1.Grid.forwardReads p.toParameters q a →
      flat (Source.scaledReads M p current Bool.false) a=flat (Source.scaledReads M p entry Bool.false) a := by
    intro a ha
    exact scaled_cell_congr M p entry current Bool.false a (same a ha)
  have stable := CrossEntropy1.Grid.forward_value_stable (Source.arithmetic M) p.toParameters
    (Source.scaledReads M p entry Bool.false) (Source.scaledReads M p current Bool.false) q scaledSame
  have logitsSame : ∀ i, current p.logits i=entry p.logits i := by
    intro i
    exact same (p.logits,i) (Or.inl rfl)
  have labelSame : CrossEntropy1.Source.label p.toParameters current q.1=
      CrossEntropy1.Source.label p.toParameters entry q.1 := by
    have h := same (Region.cast p.labels,p.labelsBase+q.1) (Or.inr rfl)
    exact congrArg (fun c => c.read .int) h
  have loadedSame : CrossEntropy1.Source.loadedLogits (Source.arithmetic M) p.toParameters
      (Source.scaledReads M p current Bool.false) q.1 q.2=
      CrossEntropy1.Source.loadedLogits (Source.arithmetic M) p.toParameters
        (Source.scaledReads M p entry Bool.false) q.1 q.2 := by
    funext lane
    unfold CrossEntropy1.Source.loadedLogits
    simp [CrossEntropy1.Source.load,Source.scaledReads,logitsSame]
  have valuesSame : Source.forwardValues M p current q.1 q.2=Source.forwardValues M p entry q.1 q.2 := by
    simp only [Source.forwardValues,CrossEntropy1.Source.forwardValues,loadedSame]
  funext a
  simp only [forwardValue,valuesSame,labelSame,stable]

theorem forward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Cell α) (a : Address) (outside : ¬forwardWrites p q a) :
    forwardStep M p q memory a=memory a := by
  by_cases hs : p.split=Bool.true
  · change Source.forward M p (unflat memory) q.1 q.2 a.1 a.2=_
    simp only [Source.forward,hs,if_true]
    rw [CrossEntropy1.Program.write_other _ _ _ _ _ _ (by
      dsimp [forwardWrites,CrossEntropy1.Grid.forwardWrites] at outside; tauto)]
    exact CrossEntropy1.Program.write_other _ _ _ _ _ _ (by
      dsimp [forwardWrites,CrossEntropy1.Grid.forwardWrites] at outside; tauto)
  · apply Source.forward_frame
    all_goals dsimp [forwardWrites,CrossEntropy1.Grid.forwardWrites] at outside
    all_goals have hf : p.split=Bool.false := Bool.eq_false_iff.mpr hs
    all_goals tauto

theorem forward_frozen {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (sep : ForwardSeparation p) :
    CrossEntropy1.Schedule.run (forwardStep M p) (tasks p.rows count) (flat entry)=
      CrossEntropy1.Schedule.run (CrossEntropy1.Schedule.put (forwardWrites p) (forwardValue M p entry)) (tasks p.rows count) (flat entry) := by
  apply CrossEntropy1.Schedule.freeze_reads (reads:=forwardReads p) (writes:=forwardWrites p) (entry:=flat entry)
  · intro q memory same
    have form := forward_form M p (unflat memory) q sep
    have values := forward_value_stable M p entry (unflat memory) q same
    funext a
    simpa only [flat_unflat,CrossEntropy1.Schedule.put,values] using congrFun form a
  · exact forward_frame M p
  · exact task_nodup _ _
  · intro q _ r _ _ a ha hw
    rcases ha with ha|ha <;> rcases hw with (hw|hw)|hw
    · exact sep.lossLogits (hw.1.symm.trans ha)
    · exact sep.lseLogits (hw.1.symm.trans ha)
    · exact sep.zLogits (hw.2.1.symm.trans ha)
    · exact sep.lossLabels (hw.1.symm.trans ha)
    · exact sep.lseLabels (hw.1.symm.trans ha)
    · exact sep.zLabels (hw.2.1.symm.trans ha)
  · intro _ _ _ _
    rfl

def forward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) : Memory α :=
  unflat (CrossEntropy1.Schedule.run (forwardStep M p) (tasks p.rows count) (flat entry))

/-- Loss and LSE keep CE1's established unique owner theorem. Z-loss uses the
same address layout and the established row-address injectivity theorem. -/
theorem forward_unique (p : Parameters) (count : Nat) (q : Task)
    (present : q∈tasks p.rows count) (sep : ForwardSeparation p)
    (r : RegionName) (base : Nat)
    (which : (r=p.loss ∧ base=p.lossBase) ∨ (r=p.lse ∧ base=p.lseBase) ∨ (r=p.zloss ∧ base=p.zlossBase)) :
    ∀ other∈tasks p.rows count, forwardWrites p other (r,base+q.2*p.rows+q.1) → other=q := by
  intro other ho hw
  have qb := (task_mem _ _ _).mp present
  have ob := (task_mem _ _ _).mp ho
  have sameIndex : base+q.2*p.rows+q.1=base+other.2*p.rows+other.1 := by
    rcases which with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩ <;> rcases hw with (hw|hw)|hw
    · exact hw.2
    · exact False.elim (sep.outputDistinct hw.1.symm)
    · exact False.elim (sep.zLoss hw.2.1.symm)
    · exact False.elim (sep.outputDistinct hw.1)
    · exact hw.2
    · exact False.elim (sep.zLse hw.2.1.symm)
    · exact False.elim (sep.zLoss hw.1)
    · exact False.elim (sep.zLse hw.1)
    · exact hw.2.2
  have unique := CrossEntropy1.Native.row_address_injective base p.rows p.rows q.2 other.2 q.1 other.1
    (Nat.le_refl _) qb.1 ob.1 sameIndex
  exact Prod.ext unique.2.symm unique.1.symm

theorem forward_output {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (q : Task) (present : q∈tasks p.rows count)
    (sep : ForwardSeparation p) (a : Address) (active : forwardWrites p q a)
    (unique : ∀ other∈tasks p.rows count, forwardWrites p other a → other=q) :
    forward M p entry count a.1 a.2=forwardValue M p entry q a := by
  simp only [forward,unflat,forward_frozen M p entry count sep]
  simpa only [present,if_true] using
    CrossEntropy1.Schedule.fixed_output (forwardWrites p) (forwardValue M p entry) (tasks p.rows count)
      (flat entry) q a active unique

theorem forward_outputs {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (q : Task) (present : q∈tasks p.rows count)
    (sep : ForwardSeparation p) :
    forward M p entry count p.loss (p.lossBase+q.2*p.rows+q.1)=
      .mk .real (Source.forwardLoss M p entry q.1 q.2 (CrossEntropy1.Source.label p.toParameters entry q.1)
        (Source.forwardValues M p entry q.1 q.2)) ∧
    forward M p entry count p.lse (p.lseBase+q.2*p.rows+q.1)=
      .mk .real (Source.forwardValues M p entry q.1 q.2).lse ∧
    (p.split=Bool.false → forward M p entry count p.zloss (p.zlossBase+q.2*p.rows+q.1)=
      .mk .real (Source.zLoss M p (CrossEntropy1.Source.label p.toParameters entry q.1)
        (Source.forwardValues M p entry q.1 q.2).lse)) := by
  refine ⟨?_,?_,?_⟩
  · have h := forward_output M p entry count q present sep (p.loss,p.lossBase+q.2*p.rows+q.1)
      (Or.inl (Or.inl ⟨rfl,rfl⟩)) (forward_unique p count q present sep _ _ (Or.inl ⟨rfl,rfl⟩))
    simpa [forwardValue,CrossEntropy1.Grid.forwardValue,Source.forwardLoss,Source.forwardValues,
      Source.label_unchanged,sep.zLoss.symm] using h
  · have h := forward_output M p entry count q present sep (p.lse,p.lseBase+q.2*p.rows+q.1)
      (Or.inl (Or.inr ⟨rfl,rfl⟩)) (forward_unique p count q present sep _ _ (Or.inr (Or.inl ⟨rfl,rfl⟩)))
    simpa [forwardValue,CrossEntropy1.Grid.forwardValue,Source.forwardValues,sep.zLse.symm,sep.outputDistinct] using h
  · intro hs
    have h := forward_output M p entry count q present sep (p.zloss,p.zlossBase+q.2*p.rows+q.1)
      (Or.inr ⟨hs,rfl,rfl⟩) (forward_unique p count q present sep _ _ (Or.inr (Or.inr ⟨rfl,rfl⟩)))
    simpa [forwardValue] using h

theorem forward_complete_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (sep : ForwardSeparation p) (r : RegionName) (i : Nat)
    (outside : ∀ q∈tasks p.rows count, ¬forwardWrites p q (r,i)) :
    forward M p entry count r i=entry r i := by
  simp only [forward,unflat,forward_frozen M p entry count sep]
  exact CrossEntropy1.Schedule.fixed_frame _ _ _ _ _ outside

abbrev backwardReads (p : Parameters) := CrossEntropy1.Grid.backwardReads p.toParameters
abbrev backwardWrites (p : Parameters) := CrossEntropy1.Grid.backwardWrites p.toParameters
abbrev BackwardSeparation (p : Parameters) := CrossEntropy1.Grid.BackwardSeparation p.toParameters

def backwardValue {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) := CrossEntropy1.Grid.backwardValue (Source.arithmetic M) p.toParameters
      (Source.scaledReads M p entry Bool.true)
def backwardStep {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Cell α) : Address → Cell α :=
  flat (Source.backward M p (unflat memory) q.1 q.2)

theorem backward_cell {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (lane : Fin p.block)
    (active : CrossEntropy1.Source.column p.toParameters q.2 lane.val<p.cols) :
    Source.backward M p entry q.1 q.2 p.gradient
      (CrossEntropy1.Source.gradientAddress p.toParameters q.1 (CrossEntropy1.Source.column p.toParameters q.2 lane.val))=
      .mk .real (Source.backwardValues M p entry q.1 q.2 lane) := by
  have unique : Function.Injective (fun i : Fin p.block => CrossEntropy1.Source.gradientAddress p.toParameters q.1
      (CrossEntropy1.Source.column p.toParameters q.2 i.val)) := by
    intro i j h
    apply Fin.ext
    dsimp [CrossEntropy1.Source.gradientAddress,CrossEntropy1.Source.column] at h
    omega
  have h := CrossEntropy1.Program.scatter_read p.gradient
    (fun i : Fin p.block => CrossEntropy1.Source.gradientAddress p.toParameters q.1 (CrossEntropy1.Source.column p.toParameters q.2 i.val))
    (fun i => Cell.mk .real (Source.backwardValues M p entry q.1 q.2 i))
    (fun i : Fin p.block => CrossEntropy1.Source.column p.toParameters q.2 i.val<p.cols) unique
    (List.ofFn (fun i : Fin p.block => i)) entry lane active
  simpa only [List.mem_ofFn,exists_apply_eq_apply,if_true] using h

theorem backward_form {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) :
    backwardStep M p q (flat entry)=CrossEntropy1.Schedule.put (backwardWrites p) (backwardValue M p entry) q (flat entry) := by
  funext a
  by_cases active : backwardWrites p q a
  · obtain ⟨lane,ha,rfl⟩ := active
    have active : backwardWrites p q (p.gradient,CrossEntropy1.Source.gradientAddress p.toParameters q.1
      (CrossEntropy1.Source.column p.toParameters q.2 lane.val)) := ⟨lane,ha,rfl⟩
    simp only [CrossEntropy1.Schedule.put,if_pos active,backwardValue,
      CrossEntropy1.Grid.backward_value_lane (Source.arithmetic M) p.toParameters
        (Source.scaledReads M p entry Bool.true) q lane ha]
    exact backward_cell M p entry q lane ha
  · simp only [CrossEntropy1.Schedule.put,if_neg active]
    apply Source.backward_frame
    intro lane ha
    have miss : a≠(p.gradient,CrossEntropy1.Source.gradientAddress p.toParameters q.1
        (CrossEntropy1.Source.column p.toParameters q.2 lane.val)) := by
      intro h
      exact active ⟨lane,ha,h⟩
    by_contra h
    push Not at h
    exact miss (Prod.ext h.1 h.2)

theorem backward_frozen {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (sep : BackwardSeparation p) :
    CrossEntropy1.Schedule.run (backwardStep M p) (tasks p.rows count) (flat entry)=
      CrossEntropy1.Schedule.run (CrossEntropy1.Schedule.put (backwardWrites p) (backwardValue M p entry)) (tasks p.rows count) (flat entry) := by
  apply CrossEntropy1.Schedule.freeze_reads (reads:=backwardReads p) (writes:=backwardWrites p) (entry:=flat entry)
  · intro q memory same
    have form := backward_form M p (unflat memory) q
    have values := CrossEntropy1.Grid.backward_value_stable (Source.arithmetic M) p.toParameters
      (Source.scaledReads M p entry Bool.true) (Source.scaledReads M p (unflat memory) Bool.true) q
      (by intro a ha; exact scaled_cell_congr M p entry (unflat memory) Bool.true a (same a ha))
    funext a
    simpa only [flat_unflat,CrossEntropy1.Schedule.put,backwardValue,values] using congrFun form a
  · intro q memory a outside
    have form := congrFun (backward_form M p (unflat memory) q) a
    simpa only [flat_unflat,CrossEntropy1.Schedule.put,if_neg outside] using form
  · exact task_nodup _ _
  · exact CrossEntropy1.Grid.backward_read_separation p.toParameters count sep
  · intro _ _ _ _
    rfl

def backward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) : Memory α :=
  unflat (CrossEntropy1.Schedule.run (backwardStep M p) (tasks p.rows count) (flat entry))

theorem backward_output {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (q : Task) (present : q∈tasks p.rows count)
    (sep : BackwardSeparation p) (lane : Fin p.block)
    (active : CrossEntropy1.Source.column p.toParameters q.2 lane.val<p.cols) :
    backward M p entry count p.gradient (CrossEntropy1.Source.gradientAddress p.toParameters q.1
      (CrossEntropy1.Source.column p.toParameters q.2 lane.val))=
      .mk .real (Source.backwardValues M p entry q.1 q.2 lane) := by
  have unique : ∀ other∈tasks p.rows count, backwardWrites p other
      (p.gradient,CrossEntropy1.Source.gradientAddress p.toParameters q.1
        (CrossEntropy1.Source.column p.toParameters q.2 lane.val)) → other=q := by
    intro other ho hw
    obtain ⟨otherLane,hactive,haddr⟩ := hw
    have same := congrArg Prod.snd haddr
    exact (CrossEntropy1.Grid.gradient_unique p.toParameters other q otherLane lane sep.rows
      ((task_mem _ _ _).mp ho).1 ((task_mem _ _ _).mp present).1 hactive active same.symm).1
  have h := CrossEntropy1.Schedule.fixed_output (backwardWrites p) (backwardValue M p entry) (tasks p.rows count)
    (flat entry) q (p.gradient,CrossEntropy1.Source.gradientAddress p.toParameters q.1
      (CrossEntropy1.Source.column p.toParameters q.2 lane.val)) ⟨lane,active,rfl⟩ unique
  simp only [backward,unflat,backward_frozen M p entry count sep]
  simpa only [present,if_true,backwardValue,
    CrossEntropy1.Grid.backward_value_lane (Source.arithmetic M) p.toParameters
      (Source.scaledReads M p entry Bool.true) q lane active] using h

theorem backward_complete_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (sep : BackwardSeparation p) (r : RegionName) (i : Nat)
    (outside : ∀ q∈tasks p.rows count, ¬backwardWrites p q (r,i)) :
    backward M p entry count r i=entry r i := by
  simp only [backward,unflat,backward_frozen M p entry count sep]
  exact CrossEntropy1.Schedule.fixed_frame _ _ _ _ _ outside

#axiomsClean forward_frozen
#axiomsClean forward_outputs
#axiomsClean forward_complete_frame
#axiomsClean backward_frozen
#axiomsClean backward_output
#axiomsClean backward_complete_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Grid
