import bench.tritonbench_g.layer_norm_fwd.Proofs.FP
import bench.tritonbench_g.layer_norm_fwd.Proofs.Domain
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Semantics.TileOps

/-! Concrete nonempty complete-source typed executions in an explicit
consistency algebra. Both normalization modes execute every optional feature,
from empty registers at PID 1, with a two-column row in a four-lane block.
This witness is not a model of IEEE arithmetic or native GPU execution; the
universal real correctness and full typed FP equivalence are separate theorems. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def model : Algebra (WithBot ℝ) where
  literal := fun _ _ value => some value
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b | .div => WithBot.realDiv a b
    | .max => max a b | .pow => some 0
  unary := fun _ op a => match op with
    | .sqrt => WithBot.realSqrt a | _ => a
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  cast := fun _ _ _ value => value
  fromNat := fun _ value => some value
  fromInt := fun _ value => some value
  fp32Bits := fun value => some (value.decodeRat.getD 0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} ax keep values =>
    ((Tile.reduceMax ax keep ⟨values⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep values => (Tile.reduceSum ax keep ⟨values⟩).data

def flags (rms : Bool) : Flags :=
  ⟨rms, Bool.true, Bool.true, Bool.true, Bool.true,
    Bool.true, Bool.true, Bool.true, Bool.true, Bool.true⟩

def regions : Regions :=
  ⟨"X", "Y", "W", "B", "RESIDUAL", "X1", "W1", "B1", "Y1",
    "RESIDUAL_OUT", "ROWSCALE", "SEEDS", "DROPOUT_MASK", "Mean", "Rstd"⟩

def strides : Strides := ⟨3, 4, 5, 6, 7, 8⟩

theorem legal_block : Random.ArangeSizeDomain 4 :=
  ⟨by decide, by decide, ⟨2, by decide⟩⟩

/-- Negative signed seed cells exercise the real int64 ABI and the modular
uint32 cast; every floating input is zero and every register starts empty. -/
def initial : State (WithBot ℝ) where
  mem := fun region _ => if region = "SEEDS" then .mk .int (-1) else .mk .real (some 0)
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun dtype _ _ => defaultValue dtype

theorem seed_valid :
    Domain.randomProfile.inputsValid (shape := [4]) (Formula.seedU32 (-1))
      (fun j => (j.1.val : Int)) = Bool.true :=
  Domain.random_inputs_valid (-1) legal_block

theorem seed_valid_any (seed : Int) :
    (⟨.u32, .i32, 7⟩ : Random.SourceProfile).inputsValid (shape := [4])
      ((seed % (2^32 : Int)).toNat : Int) (fun j => (j.1.val : Int)) = Bool.true := by
  rw [Int.toNat_of_nonneg (Int.emod_nonneg seed (by norm_num : (2^32 : Int) ≠ 0))]
  exact Domain.random_inputs_valid seed legal_block

theorem compare_available (precision : Option ComputeDType) (dtype : FloatDType) :
    (engine model).compareLt precision dtype = some (fun a b => decide (a < b)) := rfl

/-- Control registers that later source stages read. Floating register
values remain existential, so successful execution does not require evaluating
or simplifying the canonical RNG, eager division, or reductions. -/
def Ready (s : State (WithBot ℝ)) : Prop :=
  s.regs .nat [] "row" = some (fun _ => 1) ∧
  s.regs .nat [4] "cols" = some (fun j => j.1.val) ∧
  s.regs .ptr [] "Y" = some (fun _ => (("Y" : RegionName), 4)) ∧
  s.regs .ptr [] "X1" = some (fun _ => (("X1" : RegionName), 7)) ∧
  s.regs .ptr [] "RESIDUAL" = some (fun _ => (("RESIDUAL" : RegionName), 5)) ∧
  s.regs .ptr [] "RESIDUAL_OUT" = some (fun _ => (("RESIDUAL_OUT" : RegionName), 6)) ∧
  s.regs .ptr [] "Y1" = some (fun _ => (("Y1" : RegionName), 8)) ∧
  ∃ x, s.regs .real [4] "x" = some x

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

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
