import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerWords

/-! Finite input-word relation composed with the complete original integer grid.
Both inputs are constrained below N, including full tags. Logical aliases are
allowed when their word/cell relations agree. No physical-byte alias claim. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWords
open VeriTile.Triton FP.Structural
open IntegerSource (Canonical)
open ValueProofs (Parameters)
noncomputable section

def InputWords {α : Type} (d : Canonical) (words : Nat → Word d)
    (region : RegionName) (N : Nat) (mem : Memory α) : Prop :=
  ∀ o, o < N → mem region o = decodeCell d (words o)

def wordSpec {α : Type} (x y : Canonical) (wx : Nat → Word x) (wy : Nat → Word y)
    (p : Parameters) (initial : Memory α) : Memory α :=
  fun r o => if r = p.output ∧ o < p.n then
    decodeCell x (resultWord x y (wx o) (wy o)) else initial r o

theorem reference_refines {α : Type} [Inhabited α] (x y : Canonical)
    (wx : Nat → Word x) (wy : Nat → Word y) (p : Parameters) (initial : State α)
    (hx : InputWords x wx p.x p.n initial.mem) (hy : InputWords y wy p.y p.n initial.mem) :
    IntegerValueCorrect.spec x y p initial = wordSpec x y wx wy p initial.mem := by
  funext r o
  by_cases active : r = p.output ∧ o < p.n
  · simp only [IntegerValueCorrect.spec, wordSpec, active]
    exact result_refines x y (wx o) (wy o) p initial.mem o (hx o active.2) (hy o active.2)
  · simp only [IntegerValueCorrect.spec, wordSpec, active, ↓reduceIte]

theorem wordSpec_frame {α : Type} (x y : Canonical)
    (wx : Nat → Word x) (wy : Nat → Word y) (p : Parameters) (initial : Memory α)
    (r : RegionName) (o : Nat) (outside : r ≠ p.output ∨ p.n ≤ o) :
    wordSpec x y wx wy p initial r o = initial r o := by
  unfold wordSpec
  rw [if_neg]
  intro active
  rcases outside with hr | hn
  · exact hr active.1
  · omega

theorem wordSpec_readback {α : Type} [Inhabited α] (x y : Canonical)
    (wx : Nat → Word x) (wy : Nat → Word y) (p : Parameters) (initial : Memory α)
    (o : Nat) (active : o < p.n) :
    encodeCell x (wordSpec x y wx wy p initial p.output o) = resultWord x y (wx o) (wy o) := by
  simp only [wordSpec, active, and_self, ↓reduceIte, storage_roundtrip]

/-- Construct both input regions from words. When their names coincide, the
two requested decoded cells must agree, as stated by withInputs_y below.
No distinctness guard is added to the original source. -/
def withInputs {α : Type} (x y : Canonical)
    (wx : Nat → Word x) (wy : Nat → Word y) (p : Parameters) (initial : State α) : State α :=
  { initial with mem := fun r o =>
      if r = p.x ∧ o < p.n then decodeCell x (wx o)
      else if r = p.y ∧ o < p.n then decodeCell y (wy o) else initial.mem r o }

theorem withInputs_x {α : Type} (x y : Canonical)
    (wx : Nat → Word x) (wy : Nat → Word y) (p : Parameters) (initial : State α) :
    InputWords x wx p.x p.n (withInputs x y wx wy p initial).mem := by
  intro o ho
  simp [withInputs, ho]

theorem withInputs_y {α : Type} (x y : Canonical)
    (wx : Nat → Word x) (wy : Nat → Word y) (p : Parameters) (initial : State α)
    (compatible : p.y = p.x → ∀ o, o < p.n → decodeCell (α := α) x (wx o) = decodeCell y (wy o)) :
    InputWords y wy p.y p.n (withInputs x y wx wy p initial).mem := by
  intro o ho
  by_cases sameRegion : p.y = p.x
  · simpa [withInputs, ho, sameRegion] using compatible sameRegion o ho
  · simp [withInputs, ho, sameRegion]

#specNonCircular wordSpec avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean reference_refines
#axiomsClean wordSpec_frame
#axiomsClean wordSpec_readback
#axiomsClean withInputs_x
#axiomsClean withInputs_y
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWords
