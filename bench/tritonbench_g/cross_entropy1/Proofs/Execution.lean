import bench.tritonbench_g.cross_entropy1.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropy1.Program
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

@[simp] theorem write_same {α : Type} (memory : Memory α) (region : RegionName)
    (offset : Nat) (cell : Source.Cell α) : write memory region offset cell region offset=cell := by
  simp [write]

theorem write_other {α : Type} (memory : Memory α) (region other : RegionName)
    (offset index : Nat) (cell : Source.Cell α) (different : other≠region ∨ index≠offset) :
    write memory region offset cell other index=memory other index := by
  simp only [write]
  split
  · rename_i h
    rcases different with hne|hne <;> simp_all
  · rfl

theorem forwardLoss_input {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (before after : Memory α) (same : ∀ i, after p.logits i=before p.logits i)
    (row block : Nat) (target : Int) (values : ForwardValues α p.block) :
    forwardLoss M p after row block target values=forwardLoss M p before row block target values := by
  simp [forwardLoss,load,same]

/-- Both actual scalar stores are connected to initial input memory. The only
separation conditions come from the two ordinary fresh host allocations. -/
theorem forward_outputs {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (outputs : p.lse≠p.loss) (input : p.lse≠p.logits) :
    (Source.forward M p memory row block p.lse (p.lseBase+block*p.rows+row)).read .real=
      (forwardValues M p memory row block).lse ∧
    (Source.forward M p memory row block p.loss (p.lossBase+block*p.rows+row)).read .real=
      forwardLoss M p memory row block (Source.label p memory row) (forwardValues M p memory row block) := by
  have stable : ∀ i, write memory p.lse (p.lseBase+block*p.rows+row)
      (.mk .real (forwardValues M p memory row block).lse) p.logits i=memory p.logits i := by
    intro i
    exact write_other _ _ _ _ _ _ (Or.inl input.symm)
  constructor
  · simp [Source.forward,write,outputs]
  · simp only [Source.forward,write_same,Source.Cell.read_same]
    exact forwardLoss_input M p memory _ stable row block _ _

/-- Every tagged cell outside the two actual forward stores is unchanged. -/
theorem forward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (r : RegionName) (i : Nat)
    (lossMiss : r≠p.loss ∨ i≠p.lossBase+block*p.rows+row)
    (lseMiss : r≠p.lse ∨ i≠p.lseBase+block*p.rows+row) :
    Source.forward M p memory row block r i=memory r i := by
  unfold Source.forward
  rw [write_other _ _ _ _ _ _ lossMiss,write_other _ _ _ _ _ _ lseMiss]

/-- Ignored rows select FP32 zero after the source probability/correction
computation. This equality is structural and includes NaN/inf input words. -/
theorem backward_ignored {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (ignored : Source.label p memory row=p.ignored)
    (lane : Fin p.block) : backwardValues M p memory row block lane=fp M 0 := by
  simp [backwardValues,ignored]


theorem scatter_frame {α ι : Type} (region : RegionName)
    (address : ι → Nat) (values : ι → Source.Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (memory : Memory α) (r : RegionName) (offset : Nat)
    (outside : ∀ i∈lanes, active i → r≠region ∨ offset≠address i) :
    (lanes.foldl (fun current i => if active i then write current region (address i) (values i)
      else current) memory) r offset=memory r offset := by
  induction lanes generalizing memory with
  | nil => rfl
  | cons i tail ih =>
    rw [List.foldl_cons]
    have rest : ∀ j∈tail, active j → r≠region ∨ offset≠address j := by
      intro j hj ha
      exact outside j (List.mem_cons_of_mem _ hj) ha
    rw [ih _ rest]
    by_cases ha : active i
    · simp only [if_pos ha]
      exact write_other _ _ _ _ _ _ (outside i (by simp) ha)
    · simp only [if_neg ha]

theorem scatter_read {α ι : Type} [DecidableEq ι] (region : RegionName)
    (address : ι → Nat) (values : ι → Source.Cell α) (active : ι → Prop) [DecidablePred active]
    (injective : Function.Injective address) (lanes : List ι) (memory : Memory α)
    (selected : ι) (selectedActive : active selected) :
    (lanes.foldl (fun current i => if active i then write current region (address i) (values i)
      else current) memory) region (address selected)=
        if selected∈lanes then values selected else memory region (address selected) := by
  induction lanes generalizing memory with
  | nil => simp
  | cons i tail ih =>
    rw [List.foldl_cons,ih]
    by_cases present : selected∈tail
    · simp [present]
    · by_cases same : selected=i
      · subst i
        simp [present,selectedActive,write_same]
      · have different : address selected≠address i := fun h => same (injective h)
        simp only [List.mem_cons, same, present, or_self, if_false]
        by_cases ha : active i
        · simp only [if_pos ha]
          exact write_other _ _ _ _ _ _ (Or.inr different)
        · simp only [if_neg ha]

/-- All source reads are evaluated before the sole masked gradient store.
This statement therefore permits the source's in-place logits/output alias. -/
theorem backward_output {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (lane : Fin p.block)
    (active : column p block lane.val<p.cols) :
    (Source.backward M p memory row block p.gradient
      (gradientAddress p row (column p block lane.val))).read .real=
        backwardValues M p memory row block lane := by
  have unique : Function.Injective (fun i : Fin p.block => gradientAddress p row (column p block i.val)) := by
    intro i j h
    apply Fin.ext
    dsimp [gradientAddress,column] at h
    omega
  have result := scatter_read p.gradient
    (fun i : Fin p.block => gradientAddress p row (column p block i.val))
    (fun i => Source.Cell.mk .real (backwardValues M p memory row block i))
    (fun i : Fin p.block => column p block i.val<p.cols) unique
    (List.ofFn (fun i : Fin p.block => i)) memory lane active
  have present : lane∈List.ofFn (fun i : Fin p.block => i) := by simp
  simp only [present,if_true] at result
  change ((List.ofFn (fun i : Fin p.block => i)).foldl _ memory p.gradient _).read .real=_
  rw [result,Source.Cell.read_same]

theorem backward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (r : RegionName) (offset : Nat)
    (outside : ∀ i : Fin p.block, column p block i.val<p.cols →
      r≠p.gradient ∨ offset≠gradientAddress p row (column p block i.val)) :
    Source.backward M p memory row block r offset=memory r offset := by
  apply scatter_frame
  intro i _ ha
  exact outside i ha

#axiomsClean forward_outputs
#axiomsClean forward_frame
#axiomsClean backward_ignored
#axiomsClean backward_output
#axiomsClean backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Program
