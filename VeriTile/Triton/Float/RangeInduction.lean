import VeriTile.Triton.Float.Structural

/-! Transport an accumulator invariant through a structural range loop. The
fold equation lets callers retain their own mathematical accumulator definitions. -/

namespace VeriTile.Triton.FP.Structural

/-- Carry a value recurrence through a loop with a natural counter. -/
theorem range_fold {α : Type} {β : Sort u} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (fold : Nat → β → β)
    (hfold : ∀ n a, fold n a =
      if stride = 0 then a else if n < stop then fold (n + stride) (update n a) else a)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg idx .nat [] (fun _ => n)) = some u ∧
        I (n + stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t) (h : I cur initial s) :
    ∃ final, range M idx cur stop stride body s = some final ∧
      Done (fold cur initial) final := by
  by_cases hz : stride = 0
  · exact ⟨s, by simp [range, hz],
      by simpa only [hfold cur initial, hz, if_true] using hdone cur initial s h⟩
  by_cases hc : cur < stop
  · obtain ⟨next, hrun, hnext⟩ := hstep cur initial s h hc
    obtain ⟨final, hfinal, hresult⟩ := range_fold M idx body update I Done
      (cur + stride) stop stride (update cur initial) next fold hfold hstep hdone hnext
    refine ⟨final, ?_, ?_⟩
    · rw [range]
      simpa only [hz, hc, if_false, if_true, hrun, Option.bind_some] using hfinal
    · simpa only [hfold cur initial, hz, hc, if_false, if_true] using hresult
  · exact ⟨s, by simp [range, hz, hc],
      by simpa only [hfold cur initial, hz, hc, if_false] using hdone cur initial s h⟩
termination_by stop - cur
decreasing_by omega

/-- Carry a natural recurrence through a signed counter starting at a
nonnegative index. The body receives the Int counter and the signed bound
is tested as-is, including when it is negative. -/
theorem rangeInt_fold {α : Type} {β : Sort u} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur : Nat) (stop : Int) (stride : Nat) (initial : β) (s : State α)
    (fold : Nat → β → β)
    (hfold : ∀ n a, fold n a =
      if stride = 0 then a else if n < stop.toNat then fold (n + stride) (update n a) else a)
    (hstep : ∀ n a t, I n a t → n < stop.toNat →
      ∃ u, run M body (t.setReg idx .int [] (fun _ => (n : Int))) = some u ∧
        I (n + stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t) (h : I cur initial s) :
    ∃ final, rangeInt M idx (cur : Int) stop stride body s = some final ∧
      Done (fold cur initial) final := by
  by_cases hz : stride = 0
  · exact ⟨s, by simp [rangeInt, hz],
      by simpa only [hfold cur initial, hz, if_true] using hdone cur initial s h⟩
  by_cases hc : cur < stop.toNat
  · have hi : (cur : Int) < stop := by omega
    obtain ⟨next, hrun, hnext⟩ := hstep cur initial s h hc
    obtain ⟨final, hfinal, hresult⟩ := rangeInt_fold M idx body update I Done
      (cur + stride) stop stride (update cur initial) next fold hfold hstep hdone hnext
    refine ⟨final, ?_, ?_⟩
    · rw [rangeInt]
      simpa only [hz, hi, if_false, if_true, hrun, Option.bind_some, Nat.cast_add] using hfinal
    · simpa only [hfold cur initial, hz, hc, if_false, if_true] using hresult
  · have hi : ¬ (cur : Int) < stop := by omega
    exact ⟨s, by simp [rangeInt, hz, hi],
      by simpa only [hfold cur initial, hz, hc, if_false] using hdone cur initial s h⟩
termination_by stop.toNat - cur
decreasing_by omega

end VeriTile.Triton.FP.Structural
