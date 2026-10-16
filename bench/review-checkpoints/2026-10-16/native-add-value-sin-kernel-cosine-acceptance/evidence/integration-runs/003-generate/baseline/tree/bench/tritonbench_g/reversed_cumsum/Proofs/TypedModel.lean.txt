import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

/-! A concrete consistency backend for successful typed executions. Dot and
sum use their mathematical tile operations; casts are identities. This is
explicitly not an IEEE rounding model or a native GPU execution claim. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical

def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ _ a => a
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
  dot := fun _ {batch} {_} {_} {_} a b => some (Tile.dot batch ⟨a⟩ ⟨b⟩).data
  dotWithOptions := fun _ _ {batch} {_} {_} {_} a b => some (Tile.dot batch ⟨a⟩ ⟨b⟩).data
  dotTyped := fun _ left right out _ {batch} {_} {_} {_} a b =>
    some (fun i => ofFloat out ((Tile.dot batch ⟨fun j => toFloat left (a j)⟩
      ⟨fun j => toFloat right (b j)⟩).data i))

def paddingValue : (d : TileDType) → Value (WithBot ℝ) d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => some 7
  | d => defaultValue d

/-- Empty registers, nonzero feature/head IDs and nonzero unspecified tail.
The witness uses a five-row column in three two-row chunks, so the first
executed chunk has a masked time lane and every chunk has a feature tail. -/
def initial : State (WithBot ℝ) where
  mem := fun r _ => if r = "s" then .mk .bf16 (some 2) else .mk .fp16 (some (-9))
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => paddingValue d

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
end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution
