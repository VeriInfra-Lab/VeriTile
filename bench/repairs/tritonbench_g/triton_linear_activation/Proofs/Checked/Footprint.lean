import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Autotune

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
set_option maxHeartbeats 6000000

def Event.Safe (allowed : Mode → Address → Prop) : Event α β → Prop
  | .store mode p _ => allowed mode p
  | _ => True

def Code.Safe (allowed : Mode → Address → Prop) : Code α β → Prop
  | .pure _ => True
  | .event e next => Event.Safe allowed e ∧ ∀ x, Code.Safe allowed (next x)

theorem safe_bind (c : Code α β) (f : β → Code α γ)
    (hc : Code.Safe allowed c) (hf : ∀ x, Code.Safe allowed (f x)) :
    Code.Safe allowed (c.bind f) := by
  induction c with
  | pure x => exact hf x
  | event e next ih => exact ⟨hc.1,fun x => ih x f (hc.2 x) hf⟩

theorem safe_emit (e : Event α β) (h : Event.Safe allowed e) : Code.Safe allowed (emit e) :=
  ⟨h,fun _ => trivial⟩

theorem safe_gather (f : Fin n → Code α α) (h : ∀ i, Code.Safe allowed (f i)) :
    Code.Safe allowed (gatherCode f) := by
  induction n with
  | zero => trivial
  | succ n ih =>
    apply safe_bind _ _ (h 0)
    intro x
    exact safe_bind _ _ (ih (fun i => f i.succ) (fun i => h i.succ)) (fun _ => trivial)

theorem safe_fold (f : Nat → α → Code α α) (h : ∀ i x, Code.Safe allowed (f i x)) (n : Nat) (z : α) :
    Code.Safe allowed (foldCode f n z) := by
  induction n with
  | zero => trivial
  | succ n ih => exact safe_bind _ _ ih (h n)

theorem safe_operand (mode : Mode) (K bk block lane buffer row : Nat) :
    Code.Safe allowed (operandCode (α:=α) mode K bk block lane buffer row) := by
  unfold operandCode
  split <;> exact safe_emit _ trivial

theorem safe_block (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (m n block : Nat) (z : α) :
    Code.Safe allowed (blockCode s b c mode m n block z) := by
  apply safe_fold
  intro group seed
  apply safe_bind _ _ (safe_gather _ (fun _ => safe_operand _ _ _ _ _ _ _))
  intro xs
  apply safe_bind _ _ (safe_gather _ (fun _ => safe_operand _ _ _ _ _ _ _))
  intro ys
  exact safe_emit _ trivial

theorem safe_accumulation (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (m n : Nat) :
    Code.Safe allowed (accumulationCode (α:=α) s b c mode m n) := by
  cases bias : b.bias <;>
    simp only [accumulationCode,bias,bind,Code.bind,emit,Code.Safe,Event.Safe,true_and,pure]
  all_goals repeat intro
  all_goals exact safe_fold _ (fun _ _ => safe_block _ _ _ _ _ _ _ _) _ _

theorem safe_activation (name : Option String) (x : α) :
    Code.Safe allowed (activationCode name x) := by
  unfold activationCode
  split_ifs <;> simp [bind,Code.bind,emit,Code.Safe,Event.Safe,pure]

theorem safe_lane (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (name : Option String) (i : Nat)
    (out : allowed mode (b.out,i)) (saved : ∀ p, b.saved = some p → allowed mode (p,i)) :
    Code.Safe allowed (laneCode (α:=α) s b c mode name i) := by
  apply safe_bind _ _ (safe_accumulation _ _ _ _ _ _)
  intro z
  cases hs : b.saved
  · change Code.Safe allowed ((activationCode name z).bind (fun y => emit (.store mode (b.out,i) y)))
    exact safe_bind _ _ (safe_activation _ _) (fun _ => safe_emit _ out)
  · simp only [hs]
    exact safe_bind _ _ (safe_emit _ (saved _ hs)) (fun _ =>
      safe_bind _ _ (safe_activation _ _) (fun _ => safe_emit _ out))

theorem safe_sequence (f : Nat → Code α Unit) (tasks : List Nat)
    (h : ∀ i ∈ tasks, Code.Safe allowed (f i)) : Code.Safe allowed (sequenceCode f tasks) := by
  induction tasks with
  | nil => trivial
  | cons i rest ih =>
    exact safe_bind _ _ (h i (by simp)) (fun _ => ih (fun j hj => h j (by simp [hj])))

theorem safe_items (f : β → Code α Unit) (tasks : List β)
    (h : ∀ i ∈ tasks, Code.Safe allowed (f i)) : Code.Safe allowed (sequenceItems f tasks) := by
  induction tasks with
  | nil => trivial
  | cons i rest ih =>
    exact safe_bind _ _ (h i (by simp)) (fun _ => ih (fun j hj => h j (by simp [hj])))

theorem safe_grid (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (name : Option String)
    (out : ∀ i, i < s.m*s.n → allowed mode (b.out,i))
    (saved : ∀ p, b.saved = some p → ∀ i, i < s.m*s.n → allowed mode (p,i)) :
    Code.Safe allowed (gridCode (α:=α) s b c mode name) := by
  apply safe_sequence
  intro pid _
  apply safe_sequence
  intro i hi
  have active := List.mem_range.mp hi
  unfold atomicCode
  split
  · exact safe_lane _ _ _ _ _ i (out i active) (fun p hp => saved p hp i active)
  · trivial

theorem safe_copy (fromMode toMode : Mode) (src dst count : Nat)
    (allowedDst : ∀ i, i < count → allowed toMode (dst,i)) :
    Code.Safe allowed (copyCode (α:=α) fromMode toMode src dst count) := by
  apply safe_sequence
  intro i hi
  exact safe_bind _ _ (safe_emit _ trivial) (fun _ => safe_emit _ (allowedDst i (List.mem_range.mp hi)))

/-- Every store is either in an actual fresh allocation, or in the original
FP32 saved buffer's exact logical prefix. Mode and bound remain explicit. -/
def Allowed (h : Allocation r caller (before : Memory α)) (mode : Mode) (p : Address) : Prop :=
  (p.1 = h.storage 0 ∧ p.2 < r.batch.rows*r.n ∧ mode = r.mode) ∨
  (r.mode = .autocastFP16 ∧ ∃ i : Fin 3, p.1 = h.storage (castSlot i) ∧ p.2 < r.count (inputSlot i) ∧ mode = .autocastFP16) ∨
  (r.mode = .autocastFP16 ∧ r.hasSaved = Bool.true ∧ p.1 = h.storage 4 ∧ p.2 < r.batch.rows*r.n ∧ mode = .autocastFP16) ∨
  (r.hasSaved = Bool.true ∧ p.1 = r.storage 3 ∧ p.2 < r.batch.rows*r.n ∧ mode = .fp32)

theorem safe_prepare (h : Allocation r caller (before : Memory α)) : Code.Safe (Allowed h) h.prepareCode := by
  cases mode : r.mode
  · simp only [Allocation.prepareCode,mode,pure,Code.Safe]
  · simp only [Allocation.prepareCode,mode,CopyRegions.code]
    apply safe_items
    intro i _
    apply safe_copy
    intro j hj
    exact Or.inr (Or.inl ⟨mode,i,rfl,hj,rfl⟩)

theorem safe_autotuned_kernel (h : Allocation r caller (before : Memory α)) (trials : List (Fin 98)) (c : Fin 98) :
    Code.Safe (Allowed h) (h.autotunedKernelCode trials c) := by
  have grid (selected : Fin 98) : Code.Safe (Allowed h) (gridCode (α:=α) r.shape h.buffers selected r.mode r.activation) := by
    apply safe_grid
    · intro i hi
      exact Or.inl ⟨rfl,hi,rfl⟩
    · intro p saved i hi
      have flag : r.hasSaved = Bool.true := by
        by_contra bad
        simp [Allocation.buffers,bad] at saved
      cases mode : r.mode
      · have eq : p = r.storage 3 := by simpa [Allocation.buffers,flag,mode] using saved.symm
        subst p
        exact Or.inr (Or.inr (Or.inr ⟨flag,rfl,hi,rfl⟩))
      · have eq : p = h.storage 4 := by simpa [Allocation.buffers,flag,mode] using saved.symm
        subst p
        exact Or.inr (Or.inr (Or.inl ⟨mode,flag,rfl,hi,rfl⟩))
  unfold Allocation.autotunedKernelCode
  split
  · exact safe_bind _ _ (safe_items _ trials (fun c _ => grid c)) (fun _ => grid c)
  · trivial

theorem safe_copyback (h : Allocation r caller (before : Memory α)) : Code.Safe (Allowed h) h.copybackCode := by
  cases mode : r.mode <;> simp only [Allocation.copybackCode,mode]
  · trivial
  · split
    next saved =>
      apply safe_copy
      intro i hi
      exact Or.inr (Or.inr (Or.inr ⟨saved,rfl,hi,rfl⟩))
    next unsaved => trivial

theorem safe_autotuned (h : Allocation r caller (before : Memory α)) (trials : List (Fin 98)) (c : Fin 98) :
    Code.Safe (Allowed h) (h.autotunedCode trials c) :=
  safe_bind _ _ (safe_prepare h) (fun _ =>
    safe_bind _ _ (safe_autotuned_kernel h trials c) (fun _ => safe_copyback h))

#axiomsClean safe_autotuned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
