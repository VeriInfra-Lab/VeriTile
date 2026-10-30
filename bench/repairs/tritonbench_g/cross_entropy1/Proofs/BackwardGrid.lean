import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Grid

namespace VeriTile.Bench.Repairs.CrossEntropy1.Grid
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def backwardReads (p : Parameters) (q : Task) (a : Address) : Prop :=
  a=(Region.cast p.labels,p.labelsBase+q.1) ∨
  a=(p.upstream,p.upstreamBase+q.1*p.upstreamStride) ∨
  a=(p.lse,p.lseBase+q.1) ∨
  ∃ lane : Fin p.block, column p q.2 lane.val<p.cols ∧
    a=(p.logits,logitAddress p q.1 (column p q.2 lane.val))

def backwardWrites (p : Parameters) (q : Task) (a : Address) : Prop :=
  ∃ lane : Fin p.block, column p q.2 lane.val<p.cols ∧
    a=(p.gradient,gradientAddress p q.1 (column p q.2 lane.val))

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
    simpa only [gradientAddress,column,Nat.add_left_cancel_iff] using index.symm
  rw [same]

theorem backward_cell {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) (lane : Fin p.block) (active : column p q.2 lane.val<p.cols) :
    Source.backward M p entry q.1 q.2 p.gradient (gradientAddress p q.1 (column p q.2 lane.val))=
      .mk .real (backwardValues M p entry q.1 q.2 lane) := by
  have unique : Function.Injective (fun i : Fin p.block => gradientAddress p q.1 (column p q.2 i.val)) := by
    intro i j h
    apply Fin.ext
    dsimp [gradientAddress,column] at h
    omega
  have result := Program.scatter_read p.gradient
    (fun i : Fin p.block => gradientAddress p q.1 (column p q.2 i.val))
    (fun i => Source.Cell.mk .real (backwardValues M p entry q.1 q.2 i))
    (fun i : Fin p.block => column p q.2 i.val<p.cols) unique
    (List.ofFn (fun i : Fin p.block => i)) entry lane active
  simpa only [List.mem_ofFn,exists_apply_eq_apply,if_true] using result

theorem backward_form {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (q : Task) :
    backwardStep M p q (flat entry)=
      Schedule.put (backwardWrites p) (backwardValue M p entry) q (flat entry) := by
  funext a
  by_cases active : backwardWrites p q a
  · obtain ⟨lane,ha,rfl⟩ := active
    have active : backwardWrites p q (p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) := ⟨lane,ha,rfl⟩
    simp only [Schedule.put,if_pos active,backward_value_lane M p entry q lane ha]
    exact backward_cell M p entry q lane ha
  · simp only [Schedule.put,if_neg active]
    apply Program.backward_frame
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
  simpa only [flat_unflat,Schedule.put,if_neg outside] using form

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
      simp [loadedLogits,ha,load,he]
    · simp [loadedLogits,ha]
  simp only [flat] at labelsSame upstreamSame lseSame
  have valuesEq : backwardValues M p current q.1 q.2=backwardValues M p entry q.1 q.2 := by
    funext lane
    simp [backwardValues,loadedEq,Source.label,load,labelsSame,upstreamSame,lseSame]
  funext a
  simp only [backwardValue,valuesEq]

structure BackwardSeparation (p : Parameters) : Prop where
  labels : p.gradient≠Region.cast p.labels
  upstream : p.gradient≠p.upstream
  lse : p.gradient≠p.lse
  rows : p.rows≤1 ∨ p.cols≤p.gradientStride
  logitsAlias : p.gradient=p.logits → p.gradientBase=p.logitsBase ∧ p.gradientStride=p.logitsStride

theorem gradient_unique (p : Parameters) (q other : Task) (lane otherLane : Fin p.block)
    (rows : p.rows≤1 ∨ p.cols≤p.gradientStride) (hq : q.1<p.rows) (ho : other.1<p.rows)
    (hl : column p q.2 lane.val<p.cols) (hol : column p other.2 otherLane.val<p.cols)
    (same : gradientAddress p q.1 (column p q.2 lane.val)=
      gradientAddress p other.1 (column p other.2 otherLane.val)) : q=other ∧ lane=otherLane := by
  have rowsCols : q.1=other.1 ∧ column p q.2 lane.val=column p other.2 otherLane.val := by
    rcases rows with single|strided
    · have hr : q.1=other.1 := by omega
      refine ⟨hr,?_⟩
      dsimp [gradientAddress] at same
      rw [hr] at same
      omega
    · exact Native.row_address_injective _ _ _ _ _ _ _ strided hl hol same
  obtain ⟨blockEq,laneEq⟩ := Native.tile_lane_unique p.block q.2 other.2 lane.val otherLane.val
    lane.isLt otherLane.isLt rowsCols.2
  exact ⟨Prod.ext rowsCols.1 blockEq,Fin.ext laneEq⟩

theorem backward_read_separation (p : Parameters) (count : Nat) (separation : BackwardSeparation p) :
    ∀ q∈tasks p.rows count, ∀ other∈tasks p.rows count, q≠other →
      ∀ a, backwardReads p other a → ¬backwardWrites p q a := by
  intro q hq other ho different a read written
  obtain ⟨lane,ha,writeAddr⟩ := written
  rcases read with h|h|h|⟨readLane,hr,readAddr⟩
  · exact separation.labels (congrArg Prod.fst (writeAddr.symm.trans h))
  · exact separation.upstream (congrArg Prod.fst (writeAddr.symm.trans h))
  · exact separation.lse (congrArg Prod.fst (writeAddr.symm.trans h))
  · have regions := congrArg Prod.fst (writeAddr.symm.trans readAddr)
    obtain ⟨base,stride⟩ := separation.logitsAlias regions
    have indices := congrArg Prod.snd (writeAddr.symm.trans readAddr)
    have same : gradientAddress p q.1 (column p q.2 lane.val)=
        gradientAddress p other.1 (column p other.2 readLane.val) := by
      simpa only [gradientAddress,logitAddress,base,stride] using indices
    have unique := gradient_unique p q other lane readLane separation.rows
      ((task_mem _ _ _).mp hq).1 ((task_mem _ _ _).mp ho).1 ha hr same
    exact different unique.1

theorem backward_frozen {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : BackwardSeparation p) :
    Schedule.run (backwardStep M p) (tasks p.rows count) (flat entry)=
      Schedule.run (Schedule.put (backwardWrites p) (backwardValue M p entry))
        (tasks p.rows count) (flat entry) := by
  apply Schedule.freeze_reads (reads:=backwardReads p) (writes:=backwardWrites p) (entry:=flat entry)
  · intro q memory same
    have form := backward_form M p (unflat memory) q
    have values := backward_value_stable M p entry (unflat memory) q same
    funext a
    simpa only [flat_unflat,Schedule.put,values] using congrFun form a
  · exact backward_frame M p
  · exact task_nodup _ _
  · exact backward_read_separation p count separation
  · intro _ _ _ _
    rfl

def backward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) : Memory α :=
  unflat (Schedule.run (backwardStep M p) (tasks p.rows count) (flat entry))

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
  have output := Schedule.fixed_output (backwardWrites p) (backwardValue M p entry) (tasks p.rows count)
    (flat entry) q (p.gradient,gradientAddress p q.1 (column p q.2 lane.val)) ⟨lane,active,rfl⟩ unique
  simp only [backward,unflat,backward_frozen M p entry count separation]
  simpa only [present,if_true,backward_value_lane M p entry q lane active] using output

theorem backward_complete_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : BackwardSeparation p) (r : RegionName) (i : Nat)
    (outside : ∀ q∈tasks p.rows count, ¬backwardWrites p q (r,i)) :
    backward M p entry count r i=entry r i := by
  simp only [backward,unflat,backward_frozen M p entry count separation]
  exact Schedule.fixed_frame _ _ _ _ _ outside

/-- Every logical column has exactly one launched tile and lane. -/
theorem owner (p : Parameters) (positive : 0<p.block) (row col : Nat)
    (rowBound : row<p.rows) (colBound : col<p.cols) :
    ∃ q∈tasks p.rows (Math.blocks p.cols p.block), ∃ lane : Fin p.block,
      q.1=row ∧ column p q.2 lane.val=col := by
  have coverage := Math.blocks_cover p.cols p.block positive
  have quotient : col/p.block<Math.blocks p.cols p.block := by
    apply (Nat.div_lt_iff_lt_mul positive).mpr
    omega
  refine ⟨(row,col/p.block),(task_mem _ _ _).mpr ⟨rowBound,quotient⟩,
    ⟨col%p.block,Nat.mod_lt col positive⟩,rfl,?_⟩
  simpa only [column,Nat.mul_comm] using Nat.div_add_mod col p.block

#axiomsClean owner
#axiomsClean backward_output
#axiomsClean backward_complete_frame
#axiomsClean backward_frozen
#axiomsClean gradient_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Grid
