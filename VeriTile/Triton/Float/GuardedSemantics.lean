import VeriTile.Triton.Float.GuardedRules
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ExecutionProfile

/-! Shared observations and domains for guarded execution. These definitions
do not assume any numerical laws or select an implementation model. -/
namespace VeriTile.Triton.FP.Guarded
open Structural

abbrev Domain (α : Type) := GuardKind → α → Prop

def ScalarDomain {α : Type} (D : Domain α) (guards : List OperandGuard)
    (s : State α) : Prop :=
  ∀ g ∈ guards, ∃ v, s.regs .real [] g.operand = some v ∧ D g.kind (v PUnit.unit)

def TypedScalarDomain {α : Type} (D : Domain α) (dtype : FloatDType)
    (guards : List OperandGuard) (s : State α) : Prop :=
  ∀ g ∈ guards, ∃ v, s.regs dtype.toTileDType [] g.operand = some v ∧
    D g.kind (toFloat dtype (v PUnit.unit))

@[simp] theorem typedScalarDomain_real {α : Type} (D : Domain α)
    (guards : List OperandGuard) (s : State α) :
    TypedScalarDomain D .real guards s = ScalarDomain D guards s := rfl

def scalarOutput {α : Type} (dtype : FloatDType) (s : State α) : Option (TileIndex [] → α) :=
  (s.regs dtype.toTileDType [] "out").map (fun v i => toFloat dtype (v i))

@[simp] theorem scalarOutput_real {α : Type} (s : State α) :
    scalarOutput .real s = s.regs .real [] "out" := by
  simp [scalarOutput, toFloat]

/-- `x` is the backend's encoding of an FP32 word. The strict FP32 theory
speaks about such values only. A carrier may contain other elements, for
example a BF16/FP32 pair whose components disagree; no Triton tensor holds one. -/
def Word {α : Type} (M : Algebra α) (x : α) : Prop :=
  ∃ w : FP.Value .fp32, M.fp32Bits ⟨w.bits⟩ = x

/-- The floating values a scalar of a given dtype holds: itself on a floating
channel, none on a discrete one. Predicates on floating values are stated
through this list, so they need no case analysis on the dtype. -/
def scalarFloats {α : Type} : (dtype : TileDType) → Structural.Value α dtype → List α
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => fun x => [x]
  | .nat | .int | .bool | .ptr | .blockPtr => fun _ => []

/-- Floating channels must hold encoded FP32 words; discrete channels are free. -/
def WordValue {α : Type} (M : Algebra α) (dtype : TileDType)
    (x : Structural.Value α dtype) : Prop :=
  ∀ y ∈ scalarFloats dtype x, Word M y

/-- Every floating value a fragment can read is an encoded FP32 word:
registers, memory cells and the undefined-load oracle. Guarded laws are
required on such states only. -/
structure WordState {α : Type} (M : Algebra α) (s : State α) : Prop where
  regs : ∀ dtype shape name (v : Values α dtype shape),
    s.regs dtype shape name = some v → ∀ i, WordValue M dtype (v i)
  mem : ∀ region address, ∃ dtype value,
    s.mem region address = .mk dtype value ∧ WordValue M dtype value
  undef : ∀ dtype region address, WordValue M dtype (s.undef dtype region address)

/-- A state with no registers, discrete memory and word-valued undefined loads. -/
def wordBase {α : Type} (M : Algebra α) : State α where
  mem := fun _ _ => .mk .nat 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ =>
    match dtype with
    | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => M.fp32Bits ⟨0⟩
    | .nat => 0
    | .int => 0
    | .bool => Bool.false
    | .ptr => ("", 0)
    | .blockPtr => ⟨"", 0, [], [], [], []⟩

theorem word_fp32Bits {α : Type} (M : Algebra α) (w : FP.Value .fp32) :
    Word M (M.fp32Bits ⟨w.bits⟩) := ⟨w, rfl⟩

/-- An encoded word may sit on the `.real` channel. -/
theorem wordValue_real {α : Type} {M : Algebra α} {x : α} (hx : Word M x) :
    WordValue M .real x := fun _ hy => List.mem_singleton.mp hy ▸ hx

/-- An encoded word may sit on the `.fp32` channel. -/
theorem wordValue_fp32 {α : Type} {M : Algebra α} {x : α} (hx : Word M x) :
    WordValue M .fp32 x := fun _ hy => List.mem_singleton.mp hy ▸ hx

/-- A natural-number register holds no floating value. -/
theorem wordValue_nat {α : Type} (M : Algebra α) (n : Nat) : WordValue M .nat n :=
  fun _ hy => nomatch hy

/-- Distinct FP32 words have distinct encodings. `fp32Bits` decides which
carrier values count as FP32 words; a constant `fp32Bits` leaves a single
`Word`, and `Models` then constrains the backend on almost no state. A
backend offered as a non-vacuity witness must be faithful. -/
def FaithfulWords {α : Type} (M : Algebra α) : Prop :=
  Function.Injective (fun w : FP.Value .fp32 => M.fp32Bits ⟨w.bits⟩)

/-- Each floating value of a scalar is one of `xs`. -/
def FloatIn {α : Type} (xs : List α) (dtype : TileDType)
    (x : Structural.Value α dtype) : Prop :=
  ∀ y ∈ scalarFloats dtype x, y ∈ xs

/-- Every floating value a fragment can read from `s` is listed in `xs`. This
is a fact about the state alone. `FloatsIn.words` turns it into `WordState`
once the listed values are encoded words, so the case analysis on register
names stays out of every proof that mentions a numerical model. -/
structure FloatsIn {α : Type} (s : State α) (xs : List α) : Prop where
  regs : ∀ dtype shape name (v : Values α dtype shape),
    s.regs dtype shape name = some v → ∀ i, FloatIn xs dtype (v i)
  mem : ∀ region address, ∃ dtype value,
    s.mem region address = .mk dtype value ∧ FloatIn xs dtype value
  undef : ∀ dtype region address, FloatIn xs dtype (s.undef dtype region address)

/-- A well-typed state from a listing of its floating values. -/
theorem FloatsIn.words {α : Type} {M : Algebra α} {s : State α} {xs : List α}
    (h : FloatsIn s xs) (words : ∀ x ∈ xs, Word M x) : WordState M s where
  regs := fun dtype shape name v hv i y hy => words y (h.regs dtype shape name v hv i y hy)
  mem := fun region address =>
    let ⟨dtype, value, read, listed⟩ := h.mem region address
    ⟨dtype, value, read, fun y hy => words y (listed y hy)⟩
  undef := fun dtype region address y hy => words y (h.undef dtype region address y hy)

theorem FloatsIn.mono {α : Type} {s : State α} {xs ys : List α}
    (h : FloatsIn s xs) (sub : ∀ x ∈ xs, x ∈ ys) : FloatsIn s ys where
  regs := fun dtype shape name v hv i y hy => sub y (h.regs dtype shape name v hv i y hy)
  mem := fun region address =>
    let ⟨dtype, value, read, listed⟩ := h.mem region address
    ⟨dtype, value, read, fun y hy => sub y (listed y hy)⟩
  undef := fun dtype region address y hy => sub y (h.undef dtype region address y hy)

theorem wordBase_floats {α : Type} (M : Algebra α) :
    FloatsIn (wordBase M) [M.fp32Bits ⟨0⟩] where
  regs := fun _ _ _ _ h => by cases h
  mem := fun _ _ => ⟨.nat, 0, rfl, fun _ hy => nomatch hy⟩
  undef := fun dtype _ _ => by
    cases dtype <;> first | exact fun _ hy => hy | exact fun _ hy => nomatch hy

theorem wordBase_state {α : Type} (M : Algebra α) : WordState M (wordBase M) :=
  (wordBase_floats M).words (fun _ h => List.mem_singleton.mp h ▸ word_fp32Bits M ⟨0⟩)

/-- Binding a scalar register to a listed value keeps the listing. -/
theorem FloatsIn.setReg_scalar {α : Type} {s : State α} {xs : List α}
    (h : FloatsIn s xs) (name : RegName) (dtype' : TileDType) (x : Structural.Value α dtype')
    (hx : FloatIn xs dtype' x) :
    FloatsIn (s.setReg name dtype' [] (fun _ => x)) xs where
  regs := fun dtype shape n v hv i => by
    simp only [State.setReg] at hv
    split at hv
    · split at hv
      · rename_i hd
        subst hd
        split at hv
        · rename_i hs
          subst hs
          cases hv
          exact hx
        · cases hv
      · cases hv
    · exact h.regs dtype shape n v hv i
  mem := h.mem
  undef := h.undef

/-- One operand register over `wordBase`. -/
theorem wordBase_one {α : Type} (M : Algebra α) (n₁ : RegName) (d₁ : TileDType)
    (x₁ : Structural.Value α d₁) (h₁ : WordValue M d₁ x₁) :
    WordState M ((wordBase M).setReg n₁ d₁ [] (fun _ => x₁)) :=
  (((wordBase_floats M).mono (ys := M.fp32Bits ⟨0⟩ :: scalarFloats d₁ x₁)
      (fun _ h => List.mem_singleton.mp h ▸ List.mem_cons_self)).setReg_scalar
    n₁ d₁ x₁ (fun _ h => List.mem_cons_of_mem _ h)).words
    (fun y hy => (List.mem_cons.mp hy).elim (fun e => e ▸ word_fp32Bits M ⟨0⟩) (h₁ y))

/-- Two operand registers over `wordBase`. -/
theorem wordBase_two {α : Type} (M : Algebra α) (n₁ n₂ : RegName) (d₁ d₂ : TileDType)
    (x₁ : Structural.Value α d₁) (x₂ : Structural.Value α d₂)
    (h₁ : WordValue M d₁ x₁) (h₂ : WordValue M d₂ x₂) :
    WordState M (((wordBase M).setReg n₁ d₁ [] (fun _ => x₁)).setReg n₂ d₂ [] (fun _ => x₂)) :=
  ((((wordBase_floats M).mono
        (ys := M.fp32Bits ⟨0⟩ :: (scalarFloats d₁ x₁ ++ scalarFloats d₂ x₂))
        (fun _ h => List.mem_singleton.mp h ▸ List.mem_cons_self)).setReg_scalar
      n₁ d₁ x₁ (fun _ h => List.mem_cons_of_mem _ (List.mem_append_left _ h))).setReg_scalar
      n₂ d₂ x₂ (fun _ h => List.mem_cons_of_mem _ (List.mem_append_right _ h))).words
    (fun y hy => (List.mem_cons.mp hy).elim (fun e => e ▸ word_fp32Bits M ⟨0⟩)
      (fun h => (List.mem_append.mp h).elim (h₁ y) (h₂ y)))

/-- Three operand registers over `wordBase`. -/
theorem wordBase_three {α : Type} (M : Algebra α) (n₁ n₂ n₃ : RegName) (d₁ d₂ d₃ : TileDType)
    (x₁ : Structural.Value α d₁) (x₂ : Structural.Value α d₂) (x₃ : Structural.Value α d₃)
    (h₁ : WordValue M d₁ x₁) (h₂ : WordValue M d₂ x₂) (h₃ : WordValue M d₃ x₃) :
    WordState M ((((wordBase M).setReg n₁ d₁ [] (fun _ => x₁)).setReg n₂ d₂ []
      (fun _ => x₂)).setReg n₃ d₃ [] (fun _ => x₃)) :=
  (((((wordBase_floats M).mono
          (ys := M.fp32Bits ⟨0⟩ ::
            (scalarFloats d₁ x₁ ++ (scalarFloats d₂ x₂ ++ scalarFloats d₃ x₃)))
          (fun _ h => List.mem_singleton.mp h ▸ List.mem_cons_self)).setReg_scalar
        n₁ d₁ x₁ (fun _ h => List.mem_cons_of_mem _ (List.mem_append_left _ h))).setReg_scalar
        n₂ d₂ x₂ (fun _ h => List.mem_cons_of_mem _
          (List.mem_append_right _ (List.mem_append_left _ h)))).setReg_scalar
      n₃ d₃ x₃ (fun _ h => List.mem_cons_of_mem _
        (List.mem_append_right _ (List.mem_append_right _ h)))).words
    (fun y hy => (List.mem_cons.mp hy).elim (fun e => e ▸ word_fp32Bits M ⟨0⟩)
      (fun h => (List.mem_append.mp h).elim (h₁ y)
        (fun h => (List.mem_append.mp h).elim (h₂ y) (h₃ y))))

/-- Binding a scalar register to an encoded word keeps the state well typed. -/
theorem WordState.setReg_scalar {α : Type} {M : Algebra α} {s : State α}
    (h : WordState M s) (name : RegName) (dtype' : TileDType) (x : Structural.Value α dtype')
    (hx : WordValue M dtype' x) :
    WordState M (s.setReg name dtype' [] (fun _ => x)) where
  regs := fun dtype shape n v hv i => by
    simp only [State.setReg] at hv
    split at hv
    · split at hv
      · rename_i hd
        subst hd
        split at hv
        · rename_i hs
          subst hs
          cases hv
          exact hx
        · cases hv
      · cases hv
    · exact h.regs dtype shape n v hv i
  mem := h.mem
  undef := h.undef

/-- Binding a `.real` scalar register to an encoded word keeps the state well typed. -/
theorem WordState.setReg_real {α : Type} {M : Algebra α} {s : State α}
    (h : WordState M s) (name : RegName) (x : α) (hx : Word M x) :
    WordState M (s.setReg name .real [] (fun _ => x)) :=
  h.setReg_scalar name .real x (wordValue_real hx)

def fragmentAlgebra {α : Type} (f : GuardedFragment) (M : Algebra α) : Algebra α :=
  match f.defaultPrecision with
  | none => M
  | some p => M.withDefaultPrecision p

@[simp] theorem fragmentAlgebra_none {α : Type} (f : GuardedFragment) (M : Algebra α)
    (h : f.defaultPrecision = none) : fragmentAlgebra f M = M := by
  simp [fragmentAlgebra, h]

end VeriTile.Triton.FP.Guarded
