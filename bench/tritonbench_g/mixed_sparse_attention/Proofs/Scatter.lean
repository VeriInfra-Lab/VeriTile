import VeriTile.Triton.Float.MaskedScatter
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
      else t) s).regs = s.regs :=
  State.masked_scatter_regs xs active region offset value s

theorem masked_scatter_memory {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).mem =
      xs.foldl (fun mem i => if active i then fun r o =>
        if r = region ∧ o = offset i then value i else mem r o else mem) s.mem :=
  State.masked_scatter_memory xs active region offset value s


/-- Readback needs injectivity only among active lanes, allowing arbitrary
inactive offsets and the one-active-lane case even with a zero stride. -/
theorem masked_readback_active {α : Type} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a = off b → a = b) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i) = val i :=
  State.masked_readback_active r off val active s i hi hinj


end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Scatter
