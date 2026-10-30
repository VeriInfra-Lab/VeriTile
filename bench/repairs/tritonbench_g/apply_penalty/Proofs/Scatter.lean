import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Reused snapshot-scatter lemmas from the earlier destindex_copy_kv2 repair.
Scatter over arbitrary payload and address types. Events can be
scheduled in any order, including repeated events with the same snapshot value.
This is host/grid algebra, not a new floating interpreter. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.Scatter
noncomputable section
variable {ι κ α : Type} [DecidableEq κ]

def write (m : κ → α) (k : κ) (v : α) : κ → α :=
  fun q => if q=k then v else m q

def run (events : List ι) (key : ι → κ) (value : ι → α) (m : κ → α) : κ → α :=
  events.foldl (fun state i => write state (key i) (value i)) m

@[simp] theorem write_same (m : κ → α) (k : κ) (v : α) : write m k v k = v := by
  simp [write]

theorem frame (events : List ι) (key : ι → κ) (value : ι → α) (m : κ → α)
    (q : κ) (miss : ∀ i ∈ events, q ≠ key i) : run events key value m q = m q := by
  induction events generalizing m with
  | nil => rfl
  | cons i tail ih =>
    change run tail key value (write m (key i) (value i)) q = m q
    rw [ih _ (fun j hj => miss j (by simp [hj]))]
    simp [write,miss i (by simp)]

theorem stable (events : List ι) (key : ι → κ) (value : ι → α) (m : κ → α)
    (q : κ) (v : α) (initial : m q=v)
    (agree : ∀ i ∈ events, key i=q → value i=v) : run events key value m q=v := by
  induction events generalizing m with
  | nil => exact initial
  | cons i tail ih =>
    apply ih
    · by_cases h : q=key i
      · simp [write,h,agree i (by simp) h.symm]
      · simp [write,h,initial]
    · intro j hj
      exact agree j (by simp [hj])

theorem readback (events : List ι) (key : ι → κ) (value : ι → α) (m : κ → α)
    (i : ι) (hi : i ∈ events)
    (agree : ∀ j, key j=key i → value j=value i) :
    run events key value m (key i) = value i := by
  obtain ⟨before,after,he⟩ := List.append_of_mem hi
  rw [he]
  unfold run
  rw [List.foldl_append,List.foldl_cons]
  apply stable after key value _ (key i) (value i)
  · simp
  · intro j _ hj
    exact agree j hj

theorem readback_injective (events : List ι) (key : ι → κ) (value : ι → α)
    (m : κ → α) (i : ι) (hi : i ∈ events) (hinj : Function.Injective key) :
    run events key value m (key i) = value i :=
  readback events key value m i hi (fun j h => congrArg value (hinj h))

theorem readback_on (events : List ι) (key : ι → κ) (value : ι → α) (m : κ → α)
    (i : ι) (hi : i ∈ events)
    (agree : ∀ j ∈ events, key j=key i → value j=value i) :
    run events key value m (key i) = value i := by
  obtain ⟨before,after,he⟩ := List.append_of_mem hi
  rw [he]
  unfold run
  rw [List.foldl_append,List.foldl_cons]
  apply stable after key value _ (key i) (value i)
  · simp
  · intro j hj hkey
    exact agree j (by rw [he]; simp [hj]) hkey

/-- Whole-output copyback preserves a cell outside the changed footprint even
when the host copies untouched output coordinates through private scratch. -/
theorem unchanged (events : List ι) (key : ι → κ) (value : ι → α)
    (m : κ → α) (q : κ)
    (same : ∀ i ∈ events, key i=q → value i=m q) :
    run events key value m q=m q := stable events key value m q (m q) rfl same

#axiomsClean frame
#axiomsClean stable
#axiomsClean readback
#axiomsClean readback_injective
#axiomsClean readback_on
#axiomsClean unchanged
end
end VeriTile.Bench.Repairs.ApplyPenalty.Scatter
