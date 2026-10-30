import bench.tritonbench_g.triton_argmax.Proofs.Kernels
import bench.tritonbench_g.triton_argmax.Proofs.Mathematical_BitCorrect

namespace VeriTile.Bench.Repairs.TritonArgmax
namespace Reduction

theorem pick_mem {n : Nat} {α : Type} (gt eq : α → α → Bool)
    (xs : Fin n → α) (t : Tree n) : t.pick gt eq xs ∈ t.leaves := by
  induction t with
  | leaf i => simp [Tree.pick,Tree.leaves]
  | fork a b ha hb =>
    simp only [Tree.pick,Tree.leaves]
    split <;> simp_all

theorem eval_coupled {n : Nat} {α : Type} (gt eq : α → α → Bool)
    (xs : Fin n → α) (t : Tree n) :
    (t.eval gt eq xs).1=xs (t.eval gt eq xs).2 ∧
      (t.eval gt eq xs).2.val < n ∧ (t.eval gt eq xs).2∈t.leaves := by
  exact ⟨rfl,(t.pick gt eq xs).isLt,pick_mem gt eq xs t⟩

def Tree.Complete {n : Nat} (t : Tree n) : Prop := ∀ i,i∈t.leaves

/-- Independent ordered mathematical target. -/
def IsLeastMaximum {α : Type} [LE α] {n : Nat} (xs : Fin n → α) (j : Fin n) : Prop :=
  (∀ i,xs i ≤ xs j) ∧ (∀ i,xs i=xs j → j.val ≤ i.val)

section Order
variable {α : Type} [LinearOrder α]

def orderedGt (x y : α) : Bool := decide (y < x)
def orderedEq (x y : α) : Bool := decide (x=y)

theorem wins_order (x y : α) (i j : Nat) :
    leftWins orderedGt orderedEq x y i j = Bool.true ↔ y < x ∨ x=y ∧ i < j := by
  simp [leftWins,orderedGt,orderedEq]

theorem selected_order {n : Nat} (xs : Fin n → α) (t : Tree n) :
    (∀ i,i∈t.leaves → xs i ≤ xs (t.pick orderedGt orderedEq xs)) ∧
    (∀ i,i∈t.leaves → xs i=xs (t.pick orderedGt orderedEq xs) →
      (t.pick orderedGt orderedEq xs).val ≤ i.val) := by
  induction t with
  | leaf j => simp [Tree.pick,Tree.leaves]
  | fork a b ha hb =>
    let ia := a.pick orderedGt orderedEq xs
    let ib := b.pick orderedGt orderedEq xs
    have ha' : (∀ i,i∈a.leaves → xs i ≤ xs ia) ∧
        (∀ i,i∈a.leaves → xs i=xs ia → ia.val ≤ i.val) := ha
    have hb' : (∀ i,i∈b.leaves → xs i ≤ xs ib) ∧
        (∀ i,i∈b.leaves → xs i=xs ib → ib.val ≤ i.val) := hb
    change (∀ i,i∈a.leaves++b.leaves → xs i ≤ xs
        (if leftWins orderedGt orderedEq (xs ia) (xs ib) ia.val ib.val then ia else ib)) ∧ _
    by_cases win : leftWins orderedGt orderedEq (xs ia) (xs ib) ia.val ib.val = Bool.true
    · have order := (wins_order (xs ia) (xs ib) ia.val ib.val).mp win
      have le : xs ib ≤ xs ia := by rcases order with lt | ⟨eq,_⟩; exact lt.le; exact eq.symm.le
      have idx : xs ib=xs ia → ia.val ≤ ib.val := by
        intro eq
        rcases order with lt | ⟨_,idx⟩
        · exact False.elim ((ne_of_lt lt) eq)
        · omega
      constructor
      · intro i mem
        simp only [if_pos win]
        rcases List.mem_append.mp mem with mem | mem
        · exact ha'.1 i mem
        · exact (hb'.1 i mem).trans le
      · change ∀ i,i∈a.leaves++b.leaves →
          xs i=xs (if leftWins orderedGt orderedEq (xs ia) (xs ib) ia.val ib.val then ia else ib) →
          (if leftWins orderedGt orderedEq (xs ia) (xs ib) ia.val ib.val then ia else ib).val ≤ i.val
        simp only [if_pos win]
        intro i mem eq
        rcases List.mem_append.mp mem with mem | mem
        · exact ha'.2 i mem eq
        · have eq' : xs ib=xs ia := le_antisymm le (eq ▸ hb'.1 i mem)
          exact (idx eq').trans (hb'.2 i mem (eq.trans eq'.symm))
    · have order : ¬(xs ib < xs ia ∨ xs ia=xs ib ∧ ia.val < ib.val) := by
        simpa only [←wins_order] using win
      have le : xs ia ≤ xs ib := le_of_not_gt (fun h => order (Or.inl h))
      have idx : xs ia=xs ib → ib.val ≤ ia.val := by
        intro eq
        have : ¬ia.val < ib.val := fun h => order (Or.inr ⟨eq,h⟩)
        omega
      constructor
      · intro i mem
        simp only [if_neg win]
        rcases List.mem_append.mp mem with mem | mem
        · exact (ha'.1 i mem).trans le
        · exact hb'.1 i mem
      · change ∀ i,i∈a.leaves++b.leaves →
          xs i=xs (if leftWins orderedGt orderedEq (xs ia) (xs ib) ia.val ib.val then ia else ib) →
          (if leftWins orderedGt orderedEq (xs ia) (xs ib) ia.val ib.val then ia else ib).val ≤ i.val
        simp only [if_neg win]
        intro i mem eq
        rcases List.mem_append.mp mem with mem | mem
        · have eq' : xs ia=xs ib := le_antisymm le (eq ▸ ha'.1 i mem)
          exact (idx eq').trans (ha'.2 i mem (eq.trans eq'.symm))
        · exact hb'.2 i mem eq

theorem mathematical_maximum {n : Nat} (xs : Fin n → α) (t : Tree n)
    (complete : t.Complete) : IsLeastMaximum xs (t.pick orderedGt orderedEq xs) := by
  have h := selected_order xs t
  exact ⟨fun i => h.1 i (complete i),fun i => h.2 i (complete i)⟩

theorem least_maximum_unique {n : Nat} (xs : Fin n → α) (i j : Fin n)
    (hi : IsLeastMaximum xs i) (hj : IsLeastMaximum xs j) : i=j := by
  have e := le_antisymm (hi.1 j) (hj.1 i)
  exact Fin.ext (Nat.le_antisymm (hi.2 j e) (hj.2 i e.symm))
end Order

/-- Numerical bridge conditions concern only local scalar comparisons reached
at this tree node and its children. They never assume a reduction result. -/
def Tree.LocalAgreement {n : Nat} {α β : Type} (gt eq : α → α → Bool)
    (gt' eq' : β → β → Bool) (xs : Fin n → α) (ys : Fin n → β) : Tree n → Prop
  | .leaf _ => True
  | .fork a b =>
      a.LocalAgreement gt eq gt' eq' xs ys ∧ b.LocalAgreement gt eq gt' eq' xs ys ∧
      let i := a.pick gt eq xs
      let j := b.pick gt eq xs
      gt (xs i) (xs j)=gt' (ys i) (ys j) ∧ eq (xs i) (xs j)=eq' (ys i) (ys j)

theorem pick_agrees {n : Nat} {α β : Type} (gt eq : α → α → Bool)
    (gt' eq' : β → β → Bool) (xs : Fin n → α) (ys : Fin n → β) (t : Tree n)
    (agreement : t.LocalAgreement gt eq gt' eq' xs ys) : t.pick gt eq xs=t.pick gt' eq' ys := by
  induction t with
  | leaf _ => rfl
  | fork a b ha hb =>
    obtain ⟨la,lb,hgt,heq⟩ := agreement
    simp only [Tree.pick]
    rw [←ha la,←hb lb]
    simp only [leftWins,hgt,heq]
    rfl

#axiomsClean pick_mem
#axiomsClean eval_coupled
#axiomsClean mathematical_maximum
#axiomsClean pick_agrees

end Reduction
end VeriTile.Bench.Repairs.TritonArgmax
