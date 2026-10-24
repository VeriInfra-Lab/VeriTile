import bench.tritonbench_g.fused_recurrent_retention.Proofs.FP
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Semantics.TileOps

/-! An explicit mathematical consistency backend for nonempty typed runs.
Exp2 and sums have their real mathematical meanings; casts are identities.
This backend is not an IEEE rounding model or a native execution refinement. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical

def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b | .div => WithBot.realDiv a b
    | .max => max a b | .pow => some 0
  unary := fun _ op a => match op with
    | .exp2 => WithBot.realExp2 a | _ => a
  compareLe := fun _ _ => some (fun a b => decide (a ≤ b))
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  compareEq := fun _ _ => some (fun a b => decide (a = b))
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun b => some (b.decodeRat.getD 0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} ax keep v => ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data
  reduceSumTyped := fun _ d {_} ax keep v =>
    some (fun i => ofFloat d ((Tile.reduceSum ax keep ⟨fun j => toFloat d (v j)⟩).data i))

/-- Source registers needed by subsequent statements. Their values are produced
by execution, not supplied as precomputed numerical inputs or carries. -/
def Ready {α : Type} (p : Layout) (s : State α) : Prop :=
  (∃ a, s.regs .nat [] "i_k" = some a) ∧
  (∃ a, s.regs .nat [] "i_v" = some a) ∧
  (∃ a, s.regs .nat [] "i_bh" = some a) ∧
  (∃ a, s.regs .real [] "b_b" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_q" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_k" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_v" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_o" = some a) ∧
  (∃ a, s.regs .bool [p.BK] "mask_bk" = some a) ∧
  (∃ a, s.regs .bool [p.BV] "mask_bv" = some a) ∧
  (∃ a, s.regs .bool [p.BV,p.BK] "mask_kv" = some a) ∧
  (∃ a, s.regs .real [p.BV,p.BK] "h" = some a)

@[simp] theorem scatter_regs {α ι : Type} (items : List ι) (s : State α)
    (region : ι → RegionName) (offset : ι → Nat) (active : ι → Prop)
    [DecidablePred active] (value : ι → Cell α) :
    (items.foldl (fun t i => if active i then
      t.write (region i) (offset i) (value i) else t) s).regs = s.regs := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

#axiomsClean scatter_regs
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
