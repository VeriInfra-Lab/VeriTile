import bench.repairs.tritonbench_g.max_reduction.Proofs.ReductionModel

namespace VeriTile.Bench.Repairs.MaxReduction.RealModel
open VeriTile.Triton
noncomputable section
open Classical
set_option maxHeartbeats 2000000

/-- Independent mathematical maximum: an attained upper bound, with no
reference to a reduction program, fold order, padding, or floating-point word. -/
def IsMaximum (input : Nat → ℝ) (count : Nat) (answer : ℝ) : Prop :=
  (∃ i,i<count ∧ answer=input i) ∧ ∀ i,i<count → input i≤answer

/-- Independent mathematical dimension target; ties use the least original
logical coordinate along the normalized reduction axis. -/
def IsLeastArgmax (input : Nat → ℝ) (count : Nat) (answer : ℝ) (index : Nat) : Prop :=
  index<count ∧ answer=input index ∧ (∀ i,i<count → input i≤answer) ∧
    (∀ i, i < count → input i = answer → index ≤ i)

def dominates (p q : ℝ × Nat) : Prop := q.1≤p.1 ∧ (q.1=p.1 → p.2≤q.2)
def better (p q : ℝ × Nat) : Prop := q.1<p.1 ∨ (p.1=q.1 ∧ p.2<q.2)
def pair (p q : ℝ × Nat) : ℝ × Nat := if better p q then p else q

def join : Option (ℝ × Nat) → Option (ℝ × Nat) → Option (ℝ × Nat)
  | none,b => b
  | a,none => a
  | some a,some b => some (pair a b)

theorem dominates_refl (p : ℝ × Nat) : dominates p p := ⟨le_rfl,fun _ => le_rfl⟩
theorem dominates_trans {p q r : ℝ × Nat} (pq : dominates p q) (qr : dominates q r) :
    dominates p r := by
  refine ⟨qr.1.trans pq.1,?_⟩
  intro h
  have hq : q.1=p.1 := le_antisymm pq.1 (h ▸ qr.1)
  exact (pq.2 hq).trans (qr.2 (h.trans hq.symm))

theorem pair_left (p q : ℝ × Nat) : dominates (pair p q) p := by
  unfold pair
  split_ifs with h
  · exact dominates_refl p
  · simp only [better,not_or,not_and] at h
    refine ⟨le_of_not_gt h.1,?_⟩
    intro he
    exact Nat.le_of_not_gt (h.2 he)

theorem pair_right (p q : ℝ × Nat) : dominates (pair p q) q := by
  unfold pair
  split_ifs with h
  · rcases h with h|⟨h,hi⟩
    · exact ⟨h.le,fun eq => False.elim (ne_of_lt h eq)⟩
    · exact ⟨le_of_eq h.symm,fun _ => hi.le⟩
  · exact dominates_refl q

theorem pair_selects (p q : ℝ × Nat) : pair p q=p ∨ pair p q=q := by
  unfold pair; split_ifs <;> simp

/-- The source's strict-greater/equality/index expression is exactly `pair`.
No reduction or whole-output assumption enters the Real proof. -/
theorem pair_source (p q : ℝ × Nat) :
    pair p q = if decide (p.1>q.1) || (decide (p.1=q.1) && decide (p.2<q.2)) then p else q := by
  simp [pair,better]

structure Summary (values : Nat → Option (ℝ × Nat)) (lanes : List Nat)
    (result : Option (ℝ × Nat)) : Prop where
  selected : ∀ p,result=some p → ∃ i,i∈lanes ∧ values i=some p
  bounds : ∀ i,i∈lanes → ∀ p,values i=some p → ∃ q,result=some q ∧ dominates q p

theorem summary_tree (values : Nat → Option (ℝ × Nat)) (tree : Tree) :
    Summary values tree.leaves (tree.eval join values) := by
  induction tree with
  | leaf i =>
    constructor
    · intro p hp; exact ⟨i,by simp [Tree.leaves],hp⟩
    · intro j hj p hp
      simp only [Tree.leaves,List.mem_singleton] at hj
      subst j
      exact ⟨p,hp,dominates_refl p⟩
  | node l r il ir =>
    cases hl : l.eval join values with
    | none =>
      have noleft : ∀ i∈l.leaves,values i=none := by
        intro i hi
        cases hv : values i with
        | none => rfl
        | some p => obtain ⟨q,hq,_⟩ := il.bounds i hi p hv; rw [hl] at hq; contradiction
      constructor
      · intro p hp
        have hp' : r.eval join values=some p := by simpa [Tree.eval,hl,join] using hp
        obtain ⟨i,hi,hv⟩ := ir.selected p hp'
        exact ⟨i,List.mem_append_right _ hi,hv⟩
      · intro i hi p hp
        rcases List.mem_append.mp hi with hi|hi
        · rw [noleft i hi] at hp; contradiction
        · obtain ⟨q,hq,hd⟩ := ir.bounds i hi p hp
          exact ⟨q,by simpa [Tree.eval,hl,join] using hq,hd⟩
    | some a =>
      cases hr : r.eval join values with
      | none =>
        have noright : ∀ i∈r.leaves,values i=none := by
          intro i hi
          cases hv : values i with
          | none => rfl
          | some p => obtain ⟨q,hq,_⟩ := ir.bounds i hi p hv; rw [hr] at hq; contradiction
        constructor
        · intro p hp
          have hp' : l.eval join values=some p := by simpa [Tree.eval,hl,hr,join] using hp
          obtain ⟨i,hi,hv⟩ := il.selected p hp'
          exact ⟨i,List.mem_append_left _ hi,hv⟩
        · intro i hi p hp
          rcases List.mem_append.mp hi with hi|hi
          · obtain ⟨q,hq,hd⟩ := il.bounds i hi p hp
            exact ⟨q,by simpa [Tree.eval,hl,hr,join] using hq,hd⟩
          · rw [noright i hi] at hp; contradiction
      | some b =>
        constructor
        · intro p hp
          have hpair : pair a b=p := by simpa [Tree.eval,hl,hr,join] using hp
          rcases pair_selects a b with hs|hs
          · obtain ⟨i,hi,hv⟩ := il.selected a hl
            exact ⟨i,List.mem_append_left _ hi,by rw [←hpair,hs]; exact hv⟩
          · obtain ⟨i,hi,hv⟩ := ir.selected b hr
            exact ⟨i,List.mem_append_right _ hi,by rw [←hpair,hs]; exact hv⟩
        · intro i hi p hp
          refine ⟨pair a b,by simp [Tree.eval,hl,hr,join],?_⟩
          rcases List.mem_append.mp hi with hi|hi
          · obtain ⟨q,hq,hd⟩ := il.bounds i hi p hp
            rw [hl] at hq; cases hq
            exact dominates_trans (pair_left a b) hd
          · obtain ⟨q,hq,hd⟩ := ir.bounds i hi p hp
            rw [hr] at hq; cases hq
            exact dominates_trans (pair_right a b) hd

def indexed (tree : Tree) (input : Nat → ℝ) (count : Nat) : Option (ℝ × Nat) :=
  tree.eval join (fun i => if i<count then some (input i,i) else none)

theorem indexed_correct (input : Nat → ℝ) (count width : Nat) (positive : 0<count)
    (cover : count≤width) (plan : ReductionPlan width) :
    ∃ answer index,indexed plan.tree input count=some (answer,index) ∧
      IsLeastArgmax input count answer index := by
  let values := fun i => if i<count then some (input i,i) else none
  have hs := summary_tree values plan.tree
  have hzero : 0∈plan.tree.leaves := (plan.complete 0).mpr (positive.trans_le cover)
  obtain ⟨q,hq,_⟩ := hs.bounds 0 hzero (input 0,0) (by simp [values,positive])
  obtain ⟨j,hj,hvalue⟩ := hs.selected q hq
  have hjlt : j<count := by
    by_contra h; simp [values,h] at hvalue
  have hqvalue : q=(input j,j) := by simpa [values,hjlt] using hvalue.symm
  subst q
  refine ⟨input j,j,hq,hjlt,rfl,?_,?_⟩
  · intro i hi
    obtain ⟨q,hq',hd⟩ := hs.bounds i ((plan.complete i).mpr (hi.trans_le cover))
      (input i,i) (by simp [values,hi])
    rw [hq] at hq'; cases hq'
    exact hd.1
  · intro i hi he
    obtain ⟨q,hq',hd⟩ := hs.bounds i ((plan.complete i).mpr (hi.trans_le cover))
      (input i,i) (by simp [values,hi])
    rw [hq] at hq'; cases hq'
    exact hd.2 he

/-- Plain maximum interpreted in the Real projection uses bottom solely for
masked lanes. It makes no claim that hardware -infinity denotes real zero. -/
def valueReduction (tree : Tree) (values : Nat → WithBot ℝ) : WithBot ℝ :=
  tree.eval max values

theorem tree_value_upper (tree : Tree) (values : Nat → WithBot ℝ)
    (i : Nat) (hi : i∈tree.leaves) : values i≤valueReduction tree values := by
  induction tree with
  | leaf j =>
    have h : i=j := by simpa [Tree.leaves] using hi
    subst i
    exact le_rfl
  | node l r il ir =>
    rcases List.mem_append.mp hi with h|h
    · exact (il h).trans (le_max_left _ _)
    · exact (ir h).trans (le_max_right _ _)

theorem tree_value_attained (tree : Tree) (values : Nat → WithBot ℝ) :
    ∃ i,i∈tree.leaves ∧ valueReduction tree values=values i := by
  induction tree with
  | leaf i => exact ⟨i,by simp [Tree.leaves],rfl⟩
  | node l r il ir =>
    rcases il with ⟨i,hi,he⟩
    rcases ir with ⟨j,hj,hjv⟩
    by_cases h : valueReduction l values≤valueReduction r values
    · exact ⟨j,List.mem_append_right _ hj,(max_eq_right h).trans hjv⟩
    · exact ⟨i,List.mem_append_left _ hi,(max_eq_left (le_of_not_ge h)).trans he⟩

#axiomsClean indexed_correct
#axiomsClean summary_tree
#axiomsClean tree_value_upper
#axiomsClean tree_value_attained
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.RealModel
