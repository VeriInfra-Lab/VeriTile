import bench.tritonbench_g.cross_entropy1.Proofs.Execution
import bench.tritonbench_g.cross_entropy1.Proofs.Native
import bench.tritonbench_g.cross_entropy1.Proofs.Math

namespace VeriTile.Bench.Repairs.CrossEntropy1.Schedule
noncomputable section

/-- A finite schedule executes against the evolving memory. -/
def run {P A V : Type} (execute : P → (A → V) → (A → V))
    (schedule : List P) (memory : A → V) : A → V :=
  schedule.foldl (fun current program => execute program current) memory

/-- Generic induction principle for source kernels. The premise is a local
read footprint, not a final-value or execution-success condition. -/
theorem freeze_reads {P A V : Type} [DecidableEq P]
    (execute fixed : P → (A → V) → (A → V))
    (reads writes : P → A → Prop) (entry : A → V)
    (readRule : ∀ p memory, (∀ a, reads p a → memory a=entry a) → execute p memory=fixed p memory)
    (frame : ∀ p memory a, ¬writes p a → execute p memory a=memory a)
    (schedule : List P) (nodup : schedule.Nodup)
    (separate : ∀ p∈schedule, ∀ q∈schedule, p≠q → ∀ a, reads q a → ¬writes p a)
    (memory : A → V) (same : ∀ p∈schedule, ∀ a, reads p a → memory a=entry a) :
    run execute schedule memory=run fixed schedule memory := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons p tail ih =>
    have head := readRule p memory (same p (by simp))
    have tailNodup := nodup.of_cons
    have sepTail : ∀ q∈tail, ∀ r∈tail, q≠r → ∀ a, reads r a → ¬writes q a := by
      intro q hq r hr hqr a ha
      exact separate q (by simp [hq]) r (by simp [hr]) hqr a ha
    have sameTail : ∀ q∈tail, ∀ a, reads q a → execute p memory a=entry a := by
      intro q hq a ha
      have different : p≠q := by
        intro he
        subst q
        exact (List.nodup_cons.mp nodup).1 hq
      rw [frame p memory a (separate p (by simp) q (by simp [hq]) different a ha)]
      exact same q (by simp [hq]) a ha
    change run execute tail (execute p memory)=run fixed tail (fixed p memory)
    rw [ih tailNodup sepTail (execute p memory) sameTail,head]

def put {P A V : Type} (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (values : P → A → V) (p : P) (memory : A → V) : A → V :=
  fun a => if writes p a then values p a else memory a

theorem fixed_frame {P A V : Type} (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (values : P → A → V) (schedule : List P) (memory : A → V) (a : A)
    (outside : ∀ p∈schedule, ¬writes p a) :
    run (put writes values) schedule memory a=memory a := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons p tail ih =>
    change run (put writes values) tail (put writes values p memory) a=memory a
    rw [ih _ (by intro q hq; exact outside q (by simp [hq]))]
    exact if_neg (outside p (by simp))

theorem fixed_output {P A V : Type} [DecidableEq P]
    (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (values : P → A → V) (schedule : List P) (memory : A → V) (p : P) (a : A)
    (active : writes p a) (unique : ∀ q∈schedule, writes q a → q=p) :
    run (put writes values) schedule memory a=
      if p∈schedule then values p a else memory a := by
  induction schedule generalizing memory with
  | nil => simp [run]
  | cons q tail ih =>
    change run (put writes values) tail (put writes values q memory) a=_
    rw [ih _ (by intro r hr hw; exact unique r (by simp [hr]) hw)]
    by_cases present : p∈tail
    · simp [present]
    · by_cases same : p=q
      · subst q
        simp [present,put,active]
      · have miss : ¬writes q a := by
          intro hw
          exact same (unique q (by simp) hw).symm
        simp [present,same,put,miss]

#axiomsClean freeze_reads
#axiomsClean fixed_output
#axiomsClean fixed_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Schedule

namespace VeriTile.Bench.Repairs.CrossEntropy1.Grid
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

abbrev Address := RegionName × Nat
abbrev Task := Nat × Nat

def flat {α : Type} (memory : Memory α) : Address → Source.Cell α := fun a => memory a.1 a.2
def unflat {α : Type} (memory : Address → Source.Cell α) : Memory α := fun r i => memory (r,i)
@[simp] theorem flat_unflat {α : Type} (memory : Address → Source.Cell α) : flat (unflat memory)=memory := rfl
@[simp] theorem unflat_flat {α : Type} (memory : Memory α) : unflat (flat memory)=memory := rfl

def tasks (rows blocks : Nat) : List Task := (List.range rows).product (List.range blocks)

theorem task_mem (rows blocks : Nat) (q : Task) : q∈tasks rows blocks ↔ q.1<rows ∧ q.2<blocks := by
  rcases q with ⟨row,block⟩
  simp [tasks]

theorem task_nodup (rows blocks : Nat) : (tasks rows blocks).Nodup := by
  exact List.Nodup.product (List.nodup_range) (List.nodup_range)

def forwardReads (p : Parameters) (_q : Task) (a : Address) : Prop :=
  a.1=p.logits ∨ a.1=Region.cast p.labels

def forwardWrites (p : Parameters) (q : Task) (a : Address) : Prop :=
  (a.1=p.loss ∧ a.2=p.lossBase+q.2*p.rows+q.1) ∨
  (a.1=p.lse ∧ a.2=p.lseBase+q.2*p.rows+q.1)

instance (p : Parameters) (q : Task) : DecidablePred (forwardWrites p q) := by
  intro a
  unfold forwardWrites
  infer_instance

def forwardValue {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (a : Address) : Source.Cell α :=
  if a.1=p.loss then
    .mk .real (forwardLoss M p entry q.1 q.2 (Source.label p entry q.1) (forwardValues M p entry q.1 q.2))
  else .mk .real (forwardValues M p entry q.1 q.2).lse

def forwardStep {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Source.Cell α) : Address → Source.Cell α :=
  flat (Source.forward M p (unflat memory) q.1 q.2)

structure ForwardSeparation (p : Parameters) : Prop where
  outputDistinct : p.lse≠p.loss
  lseLogits : p.lse≠p.logits
  lossLogits : p.loss≠p.logits
  lseLabels : p.lse≠Region.cast p.labels
  lossLabels : p.loss≠Region.cast p.labels

theorem forward_form {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (separation : ForwardSeparation p) :
    forwardStep M p q (flat entry)=
      Schedule.put (forwardWrites p) (forwardValue M p entry) q (flat entry) := by
  have stable : ∀ i, write entry p.lse (p.lseBase+q.2*p.rows+q.1)
      (.mk .real (forwardValues M p entry q.1 q.2).lse) p.logits i=entry p.logits i := by
    intro i
    exact Program.write_other _ _ _ _ _ _ (Or.inl separation.lseLogits.symm)
  have lossEq := Program.forwardLoss_input M p entry _ stable q.1 q.2
    (Source.label p entry q.1) (forwardValues M p entry q.1 q.2)
  funext a
  change write (write entry p.lse (p.lseBase+q.2*p.rows+q.1)
    (.mk .real (forwardValues M p entry q.1 q.2).lse)) p.loss (p.lossBase+q.2*p.rows+q.1)
    (.mk .real (forwardLoss M p _ q.1 q.2 _ _)) a.1 a.2=_
  simp only [unflat_flat]
  rw [lossEq]
  by_cases lossRegion : a.1=p.loss
  · have lseRegion : a.1≠p.lse := by rw [lossRegion]; exact separation.outputDistinct.symm
    simp [write,Schedule.put,forwardWrites,forwardValue,lossRegion,lseRegion,flat,separation.outputDistinct,separation.outputDistinct.symm]
  · by_cases lseRegion : a.1=p.lse
    · simp [write,Schedule.put,forwardWrites,forwardValue,lossRegion,lseRegion,flat,separation.outputDistinct,separation.outputDistinct.symm]
    · simp [write,Schedule.put,forwardWrites,forwardValue,lossRegion,lseRegion,flat,separation.outputDistinct,separation.outputDistinct.symm]

theorem forward_value_stable {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry current : Memory α) (q : Task)
    (same : ∀ a, forwardReads p q a → flat current a=flat entry a) :
    forwardValue M p current q=forwardValue M p entry q := by
  have logitsSame : ∀ i, current p.logits i=entry p.logits i := by
    intro i
    exact same (p.logits,i) (Or.inl rfl)
  have labelsSame : ∀ i, current (Region.cast p.labels) i=entry (Region.cast p.labels) i := by
    intro i
    exact same (Region.cast p.labels,i) (Or.inr rfl)
  have loadedEq : loadedLogits M p current q.1 q.2=loadedLogits M p entry q.1 q.2 := by
    funext i
    simp [loadedLogits,load,logitsSame]
  have valuesEq : forwardValues M p current q.1 q.2=forwardValues M p entry q.1 q.2 := by
    simp [forwardValues,loadedEq]
  funext a
  simp [forwardValue,valuesEq,Source.label,labelsSame,forwardLoss,load,logitsSame]

theorem forward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (q : Task) (memory : Address → Source.Cell α) (a : Address) (outside : ¬forwardWrites p q a) :
    forwardStep M p q memory a=memory a := by
  apply Program.forward_frame
  · dsimp [forwardWrites] at outside
    tauto
  · dsimp [forwardWrites] at outside
    tauto

theorem forward_frozen {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : ForwardSeparation p) :
    Schedule.run (forwardStep M p) (tasks p.rows count) (flat entry)=
      Schedule.run (Schedule.put (forwardWrites p) (forwardValue M p entry))
        (tasks p.rows count) (flat entry) := by
  apply Schedule.freeze_reads (reads:=forwardReads p) (writes:=forwardWrites p) (entry:=flat entry)
  · intro q memory same
    have form := forward_form M p (unflat memory) q separation
    have values := forward_value_stable M p entry (unflat memory) q same
    change forwardStep M p q memory=Schedule.put (forwardWrites p) (forwardValue M p entry) q memory
    funext a
    simpa only [flat_unflat,Schedule.put,values] using congrFun form a
  · exact forward_frame M p
  · exact task_nodup _ _
  · intro q _ r _ _ a ha hw
    rcases ha with ha|ha <;> rcases hw with hw|hw
    · exact separation.lossLogits (hw.1.symm.trans ha)
    · exact separation.lseLogits (hw.1.symm.trans ha)
    · exact separation.lossLabels (hw.1.symm.trans ha)
    · exact separation.lseLabels (hw.1.symm.trans ha)
  · intro _ _ _ _
    rfl

def forward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) : Memory α :=
  unflat (Schedule.run (forwardStep M p) (tasks p.rows count) (flat entry))

theorem forward_unique (p : Parameters) (count : Nat) (q : Task)
    (present : q∈tasks p.rows count) (separation : ForwardSeparation p)
    (r : RegionName) (base : Nat) (which : (r=p.loss ∧ base=p.lossBase) ∨ (r=p.lse ∧ base=p.lseBase)) :
    ∀ other∈tasks p.rows count,
      forwardWrites p other (r,base+q.2*p.rows+q.1) → other=q := by
  intro other ho hw
  have qBound := (task_mem _ _ _).mp present
  have oBound := (task_mem _ _ _).mp ho
  have sameIndex : base+q.2*p.rows+q.1=base+other.2*p.rows+other.1 := by
    rcases which with ⟨rfl,rfl⟩|⟨rfl,rfl⟩ <;> rcases hw with hw|hw
    · exact hw.2
    · exact False.elim (separation.outputDistinct hw.1.symm)
    · exact False.elim (separation.outputDistinct hw.1)
    · exact hw.2
  obtain ⟨blocksEq,rowsEq⟩ := Native.row_address_injective base p.rows p.rows
    q.2 other.2 q.1 other.1 (Nat.le_refl _) qBound.1 oBound.1 sameIndex
  exact Prod.ext rowsEq.symm blocksEq.symm

theorem forward_outputs {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (q : Task) (present : q∈tasks p.rows count)
    (separation : ForwardSeparation p) :
    forward M p entry count p.loss (p.lossBase+q.2*p.rows+q.1)=
      .mk .real (forwardLoss M p entry q.1 q.2 (Source.label p entry q.1) (forwardValues M p entry q.1 q.2)) ∧
    forward M p entry count p.lse (p.lseBase+q.2*p.rows+q.1)=
      .mk .real (forwardValues M p entry q.1 q.2).lse := by
  have loss := Schedule.fixed_output (forwardWrites p) (forwardValue M p entry) (tasks p.rows count)
    (flat entry) q (p.loss,p.lossBase+q.2*p.rows+q.1) (Or.inl ⟨rfl,rfl⟩)
    (forward_unique p count q present separation _ _ (Or.inl ⟨rfl,rfl⟩))
  have lse := Schedule.fixed_output (forwardWrites p) (forwardValue M p entry) (tasks p.rows count)
    (flat entry) q (p.lse,p.lseBase+q.2*p.rows+q.1) (Or.inr ⟨rfl,rfl⟩)
    (forward_unique p count q present separation _ _ (Or.inr ⟨rfl,rfl⟩))
  simp only [forward,unflat,forward_frozen M p entry count separation]
  constructor
  · simpa only [present,if_true,forwardValue] using loss
  · simpa only [present,if_true,forwardValue,if_neg separation.outputDistinct] using lse

theorem forward_complete_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : ForwardSeparation p) (r : RegionName) (i : Nat)
    (outside : ∀ q∈tasks p.rows count, ¬forwardWrites p q (r,i)) :
    forward M p entry count r i=entry r i := by
  simp only [forward,unflat,forward_frozen M p entry count separation]
  exact Schedule.fixed_frame _ _ _ _ _ outside

#axiomsClean forward_outputs
#axiomsClean forward_complete_frame
#axiomsClean forward_frozen
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Grid
