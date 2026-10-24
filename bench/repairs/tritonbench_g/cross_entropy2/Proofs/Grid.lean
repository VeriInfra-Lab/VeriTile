import bench.repairs.tritonbench_g.cross_entropy2.Kernels
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Grid

namespace VeriTile.Bench.Repairs.CrossEntropy2.Grid
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Grid (Address Task flat unflat tasks task_mem task_nodup)
noncomputable section
set_option maxHeartbeats 6000000

inductive Role where | loss | lse | zLoss
  deriving DecidableEq

def region (p : Parameters) : Role → RegionName
  | .loss => p.loss | .lse => p.lse | .zLoss => p.zLoss

def base (p : Parameters) : Role → Nat
  | .loss => p.lossBase | .lse => p.lseBase | .zLoss => p.zLossBase

structure Separation (p : Parameters) : Prop where
  distinct : Function.Injective (region p)
  logits : ∀ role, region p role≠p.logits
  labels : ∀ role, region p role≠Region.cast p.labels

def writes (p : Parameters) (q : Task) (a : Address) : Prop :=
  (a.1=p.loss ∧ a.2=p.lossBase+q.2*p.rows+q.1) ∨
  (a.1=p.lse ∧ a.2=p.lseBase+q.2*p.rows+q.1) ∨
  (a.1=p.zLoss ∧ a.2=p.zLossBase+q.2*p.rows+q.1)
instance (p : Parameters) (q : Task) : DecidablePred (writes p q) := by
  intro a; unfold writes; infer_instance

def roleValue {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α) (q : Task) : Role → α
  | .lse => (Source.values M p m q.1 q.2).lse
  | .loss => (lossZ M p m q.1 q.2 (VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters m q.1)
      (Source.values M p m q.1 q.2)).1
  | .zLoss => (lossZ M p m q.1 q.2 (VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters m q.1)
      (Source.values M p m q.1 q.2)).2

def value {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α) (q : Task) (a : Address) : Cell α :=
  .mk .real (roleValue M p m q (if a.1=p.loss then .loss else if a.1=p.lse then .lse else .zLoss))

def step {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (q : Task) (m : Address → Cell α) : Address → Cell α :=
  flat (Source.forward M p (unflat m) q.1 q.2)

theorem values_input {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (a b : Memory α)
    (same : ∀ i, a p.logits i=b p.logits i) (row block : Nat) :
    Source.values M p a row block=Source.values M p b row block := by
  have loaded : VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled M p.logitScale) p.toParameters a row block=
      VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled M p.logitScale) p.toParameters b row block := by
    funext i
    simp only [VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits,VeriTile.Bench.Repairs.CrossEntropy1.Source.load,same]
  simp only [Source.values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,loaded]

theorem loss_input {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (a b : Memory α)
    (same : ∀ i, a p.logits i=b p.logits i) (row block : Nat) (target : Int)
    (v : VeriTile.Bench.Repairs.CrossEntropy1.Source.ForwardValues α p.block) :
    lossZ M p a row block target v=lossZ M p b row block target v := by
  simp only [lossZ,Source.load,VeriTile.Bench.Repairs.CrossEntropy1.Source.load,same]

theorem form {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (m : Memory α) (q : Task) (sep : Separation p) :
    step M p q (flat m)=VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put (writes p) (value M p m) q (flat m) := by
  have zl : p.zLoss≠p.loss := fun h => Role.noConfusion (sep.distinct (show region p .zLoss=region p .loss from h))
  have zs : p.zLoss≠p.lse := fun h => Role.noConfusion (sep.distinct (show region p .zLoss=region p .lse from h))
  have sl : p.lse≠p.loss := fun h => Role.noConfusion (sep.distinct (show region p .lse=region p .loss from h))
  have stable : ∀ i, write m p.lse (p.lseBase+q.2*p.rows+q.1)
      (.mk .real (Source.values M p m q.1 q.2).lse) p.logits i=m p.logits i := by
    intro i
    exact VeriTile.Bench.Repairs.CrossEntropy1.Program.write_other _ _ _ _ _ _ (Or.inl (sep.logits .lse).symm)
  have pairEq := loss_input M p _ m stable q.1 q.2
    (VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters m q.1) (Source.values M p m q.1 q.2)
  funext a
  change Source.forward M p m q.1 q.2 a.1 a.2=_
  simp only [Source.forward]
  rw [pairEq]
  by_cases hl : a.1=p.loss <;> by_cases hs : a.1=p.lse <;> by_cases hz : a.1=p.zLoss
  all_goals simp_all [Source.write,VeriTile.Bench.Repairs.CrossEntropy1.Source.write,VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put,writes,value,roleValue,flat]

theorem frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (q : Task)
    (m : Address → Cell α) (a : Address) (outside : ¬writes p q a) : step M p q m a=m a := by
  have hl : ¬(a.1=p.loss ∧ a.2=p.lossBase+q.2*p.rows+q.1) := fun h => outside (Or.inl h)
  have hs : ¬(a.1=p.lse ∧ a.2=p.lseBase+q.2*p.rows+q.1) := fun h => outside (Or.inr (Or.inl h))
  have hz : ¬(a.1=p.zLoss ∧ a.2=p.zLossBase+q.2*p.rows+q.1) := fun h => outside (Or.inr (Or.inr h))
  simp only [step,Source.forward,flat,unflat,Source.write,VeriTile.Bench.Repairs.CrossEntropy1.Source.write,if_neg hl,if_neg hs,if_neg hz]

def reads (p : Parameters) (_q : Task) (a : Address) : Prop := a.1=p.logits ∨ a.1=Region.cast p.labels

theorem stable_value {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (entry current : Memory α) (q : Task)
    (same : ∀ a, reads p q a → flat current a=flat entry a) : value M p current q=value M p entry q := by
  have logitEq : ∀ i, current p.logits i=entry p.logits i := fun i => same (p.logits,i) (Or.inl rfl)
  have labelEq : ∀ i, current (Region.cast p.labels) i=entry (Region.cast p.labels) i := fun i => same (Region.cast p.labels,i) (Or.inr rfl)
  funext a
  simp only [value,roleValue,values_input M p current entry logitEq,VeriTile.Bench.Repairs.CrossEntropy1.Source.label,labelEq]
  simp only [loss_input M p current entry logitEq]

def forward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α) (count : Nat) : Memory α :=
  unflat (VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (step M p) (tasks p.rows count) (flat m))

theorem frozen {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α) (count : Nat)
    (sep : Separation p) :
    VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (step M p) (tasks p.rows count) (flat m)=
      VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put (writes p) (value M p m)) (tasks p.rows count) (flat m) := by
  apply VeriTile.Bench.Repairs.CrossEntropy1.Schedule.freeze_reads (reads:=reads p) (writes:=writes p) (entry:=flat m)
  · intro q current same
    have shape := form M p (unflat current) q sep
    have values := stable_value M p m (unflat current) q same
    funext a
    simpa only [VeriTile.Bench.Repairs.CrossEntropy1.Grid.flat_unflat,VeriTile.Bench.Repairs.CrossEntropy1.Schedule.put,values] using congrFun shape a
  · exact frame M p
  · exact task_nodup _ _
  · intro q _ other _ _ a ha hw
    rcases ha with h|h <;> rcases hw with ⟨eq,_⟩|⟨eq,_⟩|⟨eq,_⟩
    · exact sep.logits .loss (eq.symm.trans h)
    · exact sep.logits .lse (eq.symm.trans h)
    · exact sep.logits .zLoss (eq.symm.trans h)
    · exact sep.labels .loss (eq.symm.trans h)
    · exact sep.labels .lse (eq.symm.trans h)
    · exact sep.labels .zLoss (eq.symm.trans h)
  · intro _ _ _ _; rfl

theorem complete_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α)
    (count : Nat) (sep : Separation p) (r : RegionName) (i : Nat)
    (outside : ∀ q∈tasks p.rows count, ¬writes p q (r,i)) : forward M p m count r i=m r i := by
  simp only [forward,unflat,frozen M p m count sep]
  exact VeriTile.Bench.Repairs.CrossEntropy1.Schedule.fixed_frame _ _ _ _ _ outside

theorem writes_role (p : Parameters) (q : Task) (sep : Separation p) (role : Role) (i : Nat) :
    writes p q (region p role,i) ↔ i=base p role+q.2*p.rows+q.1 := by
  have zl : p.zLoss≠p.loss := fun h => Role.noConfusion (sep.distinct (show region p .zLoss=region p .loss from h))
  have zs : p.zLoss≠p.lse := fun h => Role.noConfusion (sep.distinct (show region p .zLoss=region p .lse from h))
  have sl : p.lse≠p.loss := fun h => Role.noConfusion (sep.distinct (show region p .lse=region p .loss from h))
  cases role <;> simp [writes,region,base,zl,zs,sl,Ne.symm zl,Ne.symm zs,Ne.symm sl]

theorem complete_output {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α)
    (count : Nat) (sep : Separation p) (q : Task) (present : q∈tasks p.rows count) (role : Role) :
    forward M p m count (region p role) (base p role+q.2*p.rows+q.1)=
      .mk .real (roleValue M p m q role) := by
  have active : writes p q (region p role,base p role+q.2*p.rows+q.1) := (writes_role p q sep role _).mpr rfl
  have unique : ∀ other∈tasks p.rows count, writes p other (region p role,base p role+q.2*p.rows+q.1) → other=q := by
    intro other ho hw
    have same := (writes_role p other sep role _).mp hw
    have qb := (task_mem _ _ _).mp present
    have ob := (task_mem _ _ _).mp ho
    obtain ⟨blocks,rows⟩ := VeriTile.Bench.Repairs.CrossEntropy1.Native.row_address_injective
      (base p role) p.rows p.rows q.2 other.2 q.1 other.1 (Nat.le_refl _) qb.1 ob.1 same
    exact Prod.ext rows.symm blocks.symm
  have result := VeriTile.Bench.Repairs.CrossEntropy1.Schedule.fixed_output (writes p) (value M p m) (tasks p.rows count) (flat m) q
    (region p role,base p role+q.2*p.rows+q.1) active unique
  have zl : p.zLoss≠p.loss := fun h => Role.noConfusion (sep.distinct (show region p .zLoss=region p .loss from h))
  have zs : p.zLoss≠p.lse := fun h => Role.noConfusion (sep.distinct (show region p .zLoss=region p .lse from h))
  have sl : p.lse≠p.loss := fun h => Role.noConfusion (sep.distinct (show region p .lse=region p .loss from h))
  simp only [forward,unflat,frozen M p m count sep]
  rw [result,if_pos present]
  cases role <;> simp [value,region,zl,zs,sl]

#axiomsClean complete_output

#axiomsClean form
#axiomsClean frozen
#axiomsClean complete_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Grid
