import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Domain
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
/-- A masked store leaves the complete register map intact, including lanes
whose inactive accesses retain unspecified values. -/
@[simp] theorem masked_scatter_regs {α ι : Type} (s : State α) (items : List ι)
    (address : ι → RegionName × Nat) (active : ι → Bool) (value : ι → Cell α) :
    (items.foldl (fun t i => if active i then
      t.write (address i).1 (address i).2 (value i) else t) s).regs = s.regs := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

/-- The same register frame after the evaluator simplifies a boolean mask
to its proposition. Separate region/offset functions match actual stores. -/
@[simp] theorem guarded_scatter_regs {α ι : Type} (s : State α) (items : List ι)
    (region : ι → RegionName) (offset : ι → Nat) (active : ι → Prop)
    [DecidablePred active] (value : ι → Cell α) :
    (items.foldl (fun t i => if active i then
      t.write (region i) (offset i) (value i) else t) s).regs = s.regs := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

@[simp] theorem scatter_regs {α ι : Type} (s : State α) (items : List ι)
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α) :
    (items.foldl (fun t i => t.write (region i) (offset i) (value i)) s).regs = s.regs := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    rfl

end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
