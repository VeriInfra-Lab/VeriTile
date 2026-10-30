import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Scatter
open VeriTile.Triton FP.Structural
noncomputable section
theorem scatter_memory {α ι : Type} (xs : List ι) (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => t.write region (offset i) (value i)) s).mem =
      xs.foldl (fun mem i => fun r o =>
        if r = region ∧ o = offset i then value i else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons, ih]; rfl


@[simp] theorem masked_scatter_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

theorem masked_scatter_memory {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).mem =
      xs.foldl (fun mem i => if active i then fun r o =>
        if r = region ∧ o = offset i then value i else mem r o else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i <;> simp only [hi, ↓reduceIte] <;> rfl


/-- Readback needs injectivity only among active lanes, allowing arbitrary
inactive offsets and the one-active-lane case even with a zero stride. -/
theorem masked_readback_active {α : Type} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a = off b → a = b) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i) = val i := by
  rw [State.masked_scatter_eq_filter]
  have himem : i ∈ (TileShape.allIndices shape).filter (fun j => decide (active j)) := by
    simp [TileShape.mem_allIndices, hi]
  obtain ⟨before, after, hl⟩ := List.append_of_mem himem
  have hnd := (TileShape.allIndices_nodup shape).filter (fun j => decide (active j))
  rw [hl, List.nodup_append, List.nodup_cons] at hnd
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hnd
  rw [hl, List.foldl_append, List.foldl_cons, State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    have hjmem : j ∈ (TileShape.allIndices shape).filter (fun j => decide (active j)) := by
      rw [hl]
      simp only [List.mem_append, List.mem_cons]
      exact Or.inr (Or.inr hj)
    have hjactive : active j := by simpa using (List.mem_filter.mp hjmem).2
    exact hnot ((hinj i j hi hjactive heq) ▸ hj)


end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Scatter
