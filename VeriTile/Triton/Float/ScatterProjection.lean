import VeriTile.Triton.Float.Structural

/-! Projections of a masked write fold whose destination can vary by lane. -/

namespace VeriTile.Triton.FP.Structural.State

theorem fold_writes_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

theorem fold_writes_mem {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).mem =
    xs.foldl (fun mem i => if active i then
      fun r o => if r = region i ∧ o = offset i then value i else mem r o
      else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]
      rfl
    · simp only [hi, ↓reduceIte]

theorem fold_writes_undef {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).undef = s.undef := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

end VeriTile.Triton.FP.Structural.State
