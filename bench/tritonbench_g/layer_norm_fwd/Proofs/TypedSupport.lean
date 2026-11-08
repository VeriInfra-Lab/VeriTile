import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Domain
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
/-- Nonnegative source addresses retain their natural-number projection. -/
@[simp] theorem nat_product_address (a b : Nat) :
    ((a:Int)*(b:Int)).toNat=a*b := by rw [←Nat.cast_mul]; rfl
@[simp] theorem nat_sum_address (a b : Nat) :
    ((a:Int)+(b:Int)).toNat=a+b := by rw [←Nat.cast_add]; rfl
@[simp] theorem nat_product_sum_address (a b c : Nat) :
    ((a:Int)*(b:Int)+(c:Int)).toNat=a*b+c := by
  rw [←Nat.cast_mul,←Nat.cast_add]; rfl
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
