import bench.tritonbench_g.masked_select.Proofs.TypedCopy

/-! Independent stable selection of a logical row-major sequence. This file
does not define its specification by running the scatter implementation. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.StableSelection

def count (mask : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => count mask n + if mask n then 1 else 0

def select {α : Type} (input : Nat → α) (mask : Nat → Bool) (n : Nat) : List α :=
  ((List.range n).filter mask).map input

@[simp] theorem count_zero (mask : Nat → Bool) : count mask 0 = 0 := rfl

theorem count_succ (mask : Nat → Bool) (n : Nat) :
    count mask (n + 1) = count mask n + if mask n then 1 else 0 := rfl

theorem select_succ {α : Type} (input : Nat → α) (mask : Nat → Bool) (n : Nat) :
    select input mask (n + 1) = select input mask n ++
      if mask n then [input n] else [] := by
  cases h : mask n <;> simp [select, List.range_succ, List.filter_append, h]

@[simp] theorem length_select {α : Type} (input : Nat → α) (mask : Nat → Bool) (n : Nat) :
    (select input mask n).length = count mask n := by
  induction n with
  | zero => simp [select, count]
  | succ n ih =>
    rw [select_succ, List.length_append, ih, count_succ]
    cases mask n <;> rfl

theorem count_le (mask : Nat → Bool) (n : Nat) : count mask n ≤ n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [count_succ]; split <;> omega

theorem count_mono (mask : Nat → Bool) {i j : Nat} (h : i ≤ j) :
    count mask i ≤ count mask j := by
  induction j with
  | zero =>
    have : i = 0 := by omega
    subst i
    rfl
  | succ j ih =>
    by_cases hij : i ≤ j
    · have := ih hij; rw [count_succ]; split <;> omega
    · have : i = j + 1 := by omega
      subst i
      rfl

theorem selected_strict (mask : Nat → Bool) {i j : Nat}
    (hi : mask i = Bool.true) (hij : i < j) : count mask i < count mask j := by
  have h := count_mono mask (show i + 1 ≤ j by omega)
  rw [count_succ, hi] at h
  simpa using h

theorem selected_injective (mask : Nat → Bool) {i j : Nat}
    (hi : mask i = Bool.true) (hj : mask j = Bool.true)
    (h : count mask i = count mask j) : i = j := by
  by_contra hne
  rcases lt_or_gt_of_ne hne with hij | hji
  · have := selected_strict mask hi hij; omega
  · have := selected_strict mask hj hji; omega

theorem inclusive_selected (mask : Nat → Bool) {i : Nat} (hi : mask i = Bool.true) :
    count mask (i + 1) - 1 = count mask i ∧ 0 < count mask (i + 1) := by
  simp [count_succ, hi]

theorem get_selected {α : Type} (input : Nat → α) (mask : Nat → Bool)
    {n i : Nat} (hi : i < n) (hm : mask i = Bool.true) :
    (select input mask n)[count mask i]? = some (input i) := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [select_succ]
    by_cases hin : i < n
    · rw [List.getElem?_append_left (by simpa using selected_strict mask hm hin)]
      exact ih hin
    · have hie : i = n := by omega
      subst i
      rw [hm]
      simp [length_select]

theorem selected_surjective (mask : Nat → Bool) {n k : Nat}
    (hk : k < count mask n) : ∃ i, i < n ∧ mask i = Bool.true ∧ count mask i = k := by
  induction n with
  | zero => simp at hk
  | succ n ih =>
    by_cases hkn : k < count mask n
    · obtain ⟨i, hi, hm, hc⟩ := ih hkn
      exact ⟨i, by omega, hm, hc⟩
    · rw [count_succ] at hk
      cases hm : mask n with
      | «false» => simp [hm] at hk; omega
      | «true» => exact ⟨n, by omega, hm, by simp [hm] at hk; omega⟩

theorem all_false_count (mask : Nat → Bool) {n : Nat}
    (h : ∀ i, i < n → mask i = Bool.false) : count mask n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [count_succ, h n (by omega), ih (fun i hi => h i (by omega))]

#axiomsClean get_selected
#axiomsClean selected_injective
#axiomsClean selected_surjective

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect.StableSelection
