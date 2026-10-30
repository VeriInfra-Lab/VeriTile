import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FP64Profiles

/-! Actual logical memory cells with distinct storage tags. The total function
maps each address to an optional cell: absence and wrong tags cannot become a
default word. This is a logical region/index model, not native byte memory. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.FP64Memory
open FP64TypedCore
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

inductive Cell where
  | mk (dtype : Storage) (value : Word (.storage dtype))

abbrev Memory := String → Nat → Option Cell

def readCell (d : Storage) : Option Cell → Option (Word (.storage d))
  | none => none
  | some (.mk actual word) => if h : actual = d then some (h ▸ word) else none

@[simp] theorem read_cell_same (d : Storage) (v : Word (.storage d)) :
    readCell d (some (.mk d v)) = some v := by simp [readCell]

@[simp] theorem read_cell_absent (d : Storage) : readCell d none = none := rfl

theorem read_cell_wrong_tag (d actual : Storage) (v : Word (.storage actual))
    (different : actual ≠ d) : readCell d (some (.mk actual v)) = none := by
  simp [readCell, different]

def writeCell (memory : Memory) (region : String) (index : Nat) (cell : Option Cell) : Memory :=
  fun r i => if r = region ∧ i = index then cell else memory r i

@[simp] theorem write_cell_readback (memory : Memory) (region : String) (index : Nat)
    (cell : Option Cell) : writeCell memory region index cell region index = cell := by
  simp [writeCell]

theorem write_cell_frame (memory : Memory) (region : String) (index : Nat)
    (cell : Option Cell) (r : String) (i : Nat) (outside : r ≠ region ∨ i ≠ index) :
    writeCell memory region index cell r i = memory r i := by
  simp only [writeCell]
  rw [if_neg (by intro h; rcases outside with h' | h'; exact h' h.1; exact h' h.2)]

structure Parameters (xD yD : Storage) where
  n : Nat
  x : String
  y : String
  output : String

def Separated (p : Parameters xD yD) : Prop := p.x ≠ p.output ∧ p.y ≠ p.output

def InputWords (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD)) (memory : Memory) : Prop :=
  ∀ index, index < p.n →
    memory p.x index = some (.mk xD (xs index)) ∧ memory p.y index = some (.mk yD (ys index))

/-- Equal input region names are allowed only when their complete tagged cells
agree. In particular, incompatible tags cannot be hidden by default reads. -/
theorem shared_input_compatibility (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD)) (memory : Memory)
    (inputs : InputWords p xs ys memory) (same : p.x = p.y) (i : Nat) (active : i < p.n) :
    Cell.mk xD (xs i) = Cell.mk yD (ys i) := by
  have hx := (inputs i active).1
  have hy := (inputs i active).2
  rw [same] at hx
  exact Option.some.inj (hx.symm.trans hy)

/-- Floating output needs no domain restriction; only active integer outputs
require successful direct f64→integer conversion under the chosen callbacks. -/
def OutputDefined (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD)) : Prop :=
  if xD.isFloating then True else
    ∀ i, i < p.n → (result b xD yD (xs i) (ys i)).isSome = true

theorem floating_defined (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (floating : xD.isFloating = true) : OutputDefined b p xs ys := by
  simp [OutputDefined, floating]

theorem defined_value (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (defined : OutputDefined b p xs ys) (i : Nat) (active : i < p.n) :
    ∃ v, result b xD yD (xs i) (ys i) = some v := by
  by_cases floating : xD.isFloating = true
  · cases xD <;> simp [Storage.isFloating] at floating
    all_goals simp [result, output64]
  · have h := (show ∀ j, j < p.n → (result b xD yD (xs j) (ys j)).isSome = true from
      by simpa [OutputDefined, floating] using defined) i active
    cases hv : result b xD yD (xs i) (ys i) with
    | none => simp [hv] at h
    | some v => exact ⟨v, rfl⟩

/-- Independent expected cell; it does not invoke any source/interpreter. -/
def outputCell (b : Numerics) (xD yD : Storage)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD)) (i : Nat) : Option Cell :=
  (result b xD yD (xs i) (ys i)).map (Cell.mk xD)

def referenceIndices (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (indices : List Nat) (initial : Memory) : Memory := fun r i =>
  if r = p.output ∧ i < p.n ∧ i ∈ indices then outputCell b xD yD xs ys i else initial r i

def reference (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (initial : Memory) : Memory := fun r i =>
  if r = p.output ∧ i < p.n then outputCell b xD yD xs ys i else initial r i

@[simp] theorem reference_nil (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD)) (initial : Memory) :
    referenceIndices b p xs ys [] initial = initial := by
  funext r i; simp [referenceIndices]

theorem reference_cons (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (i : Nat) (is : List Nat) (initial : Memory) :
    referenceIndices b p xs ys (i :: is) initial =
      referenceIndices b p xs ys is
        (if i < p.n then writeCell initial p.output i (outputCell b xD yD xs ys i) else initial) := by
  funext r j
  by_cases hr : r = p.output <;> by_cases hi : i < p.n <;> by_cases hj : j < p.n <;>
    by_cases he : j = i <;> by_cases hm : j ∈ is <;>
    simp [referenceIndices, writeCell, hr, hi, hj, he, hm]
  all_goals subst j; contradiction

theorem reference_append (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (left right : List Nat) (initial : Memory) :
    referenceIndices b p xs ys right (referenceIndices b p xs ys left initial) =
      referenceIndices b p xs ys (left ++ right) initial := by
  funext r i
  by_cases hr : r = p.output <;> by_cases hi : i < p.n <;>
    by_cases hl : i ∈ left <;> by_cases hu : i ∈ right <;>
    simp [referenceIndices, hr, hi, hl, hu]

theorem reference_inputs (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (indices : List Nat) (initial : Memory) (separated : Separated p)
    (inputs : InputWords p xs ys initial) :
    InputWords p xs ys (referenceIndices b p xs ys indices initial) := by
  intro i active
  simpa [referenceIndices, separated.1, separated.2] using inputs i active

theorem reference_frame (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (initial : Memory) (r : String) (i : Nat) (outside : r ≠ p.output ∨ p.n ≤ i) :
    reference b p xs ys initial r i = initial r i := by
  unfold reference
  rw [if_neg]
  rcases outside with h | h
  · intro a; exact h a.1
  · omega

#axiomsClean read_cell_same
#axiomsClean read_cell_absent
#axiomsClean read_cell_wrong_tag
#axiomsClean write_cell_readback
#axiomsClean write_cell_frame
#axiomsClean shared_input_compatibility
#axiomsClean floating_defined
#axiomsClean defined_value
#axiomsClean reference_nil
#axiomsClean reference_cons
#axiomsClean reference_append
#axiomsClean reference_inputs
#axiomsClean reference_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.FP64Memory
