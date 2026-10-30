/-
Precision-specialized identities in `experiments/floating_point/kernels.py`.

The fp32 instances of ROUND-IDEM and ACC-WIDEN contain only same-dtype
source copies in addition to their common arithmetic graph. A source copy is
not a numerical cast: `FP.cast` may canonicalize NaNs even at equal formats.
`sourceCopies` makes this distinction explicit without changing the general
structural interpreter. Every arithmetic primitive remains arbitrary.

These are typed scalar-lane ASTs and successful store executions. They model
the expression and output-store part of the probe after its common loads;
they are not a certificate for the Python frontend or an entire GPU launch.
In particular ACC-WIDEN/bf16_fp32 is not covered: INPUT_LOW is true there.
-/
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.CastExact.RoundTrip
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.FP.PrecisionIdentityLaws

open Structural

/-- Source-language conversion to the current dtype copies its operand.
All cross-dtype conversions and numerical operations remain those of `M`.
This adapter is explicit: arbitrary structural algebras need not obey it. -/
def sourceCopies {α : Type} (M : Algebra α) : Algebra α :=
  { M with cast := fun p src dst x => if src = dst then x else M.cast p src dst x }

@[simp] theorem sourceCopies_cast_same {α : Type} (M : Algebra α)
    (p : Option ComputeDType) (d : FloatDType) (x : α) :
    (sourceCopies M).cast p d d x = x := by simp [sourceCopies]

@[simp] theorem sourceCopies_cast_different {α : Type} (M : Algebra α)
    (p : Option ComputeDType) (src dst : FloatDType) (h : src ≠ dst) (x : α) :
    (sourceCopies M).cast p src dst x = M.cast p src dst x := by
  simp [sourceCopies, h]

def fp32Copy (e : Op .fp32 []) : Op .fp32 [] := .castFloat .fp32 .fp32 e

/-- The probe loads a value and immediately calls `.to(tl.float32)`.
The register denotes that common load, before its same-dtype copy. -/
def loadedFp32 (name : RegName) : Op .fp32 [] :=
  fp32Copy (.ref .fp32 [] name)

/-- Preserve each ordered fp32 addition and its explicit compute precision. -/
def fp32Add (a b : Op .fp32 []) : Op .fp32 [] :=
  .withPrecision .fp32 .fp32 (.add .fp32 .nil a b)

def roundIdemFp32Lhs : Op .fp32 [] :=
  fp32Copy (fp32Copy (fp32Copy (loadedFp32 "a")))

def roundIdemFp32Rhs : Op .fp32 [] := fp32Copy (loadedFp32 "a")

def accWidenFp32Lhs : Op .fp32 [] :=
  fp32Copy (fp32Add (fp32Copy (fp32Add (loadedFp32 "a") (loadedFp32 "b")))
    (loadedFp32 "c"))

def accWidenFp32Rhs : Op .fp32 [] :=
  fp32Add (fp32Add (loadedFp32 "a") (loadedFp32 "b")) (loadedFp32 "c")

/-- No reassociation: both sides call the identical ordered addition tree. -/
def accWidenFp32Value {α : Type} (M : Algebra α) (a b c : α) : α :=
  M.binary (some .fp32) .fp32 .add (M.binary (some .fp32) .fp32 .add a b) c

@[simp] theorem eval_fp32Copy {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (e : Op .fp32 []) (s : State α) :
    evalOp (sourceCopies M) p (fp32Copy e) s = evalOp (sourceCopies M) p e s := by
  rw [fp32Copy, evalOp_unfold]
  change (evalOp (sourceCopies M) p e s).bind (fun v => some (fun i => v i)) =
    evalOp (sourceCopies M) p e s
  cases evalOp (sourceCopies M) p e s <;> rfl

@[simp] theorem eval_loadedFp32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (name : RegName) (s : State α) :
    evalOp (sourceCopies M) p (loadedFp32 name) s = s.regs .fp32 [] name := by
  rw [loadedFp32, eval_fp32Copy, evalOp_unfold]

theorem eval_fp32Add {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (a b : Op .fp32 []) (s : State α) :
    evalOp (sourceCopies M) p (fp32Add a b) s = (do
      let av ← evalOp (sourceCopies M) (some .fp32) a s
      let bv ← evalOp (sourceCopies M) (some .fp32) b s
      pure (bop (M.binary (some .fp32) .fp32 .add) Broadcast.nil av bv)) := by
  rw [fp32Add, evalOp_unfold]
  change evalOp (sourceCopies M) (some .fp32) (.add .fp32 .nil a b) s = _
  rw [evalOp_unfold]
  rfl

theorem eval_roundIdemFp32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) :
    evalOp (sourceCopies M) p roundIdemFp32Lhs s =
      evalOp (sourceCopies M) p roundIdemFp32Rhs s := by
  simp only [roundIdemFp32Lhs, roundIdemFp32Rhs, eval_fp32Copy]

theorem eval_accWidenFp32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) :
    evalOp (sourceCopies M) p accWidenFp32Lhs s =
      evalOp (sourceCopies M) p accWidenFp32Rhs s := by
  simp only [accWidenFp32Lhs, accWidenFp32Rhs, eval_fp32Copy, eval_fp32Add]

theorem eval_roundIdemFp32_success {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (a : α)
    (ha : s.regs .fp32 [] "a" = some (fun _ => a)) :
    evalOp (sourceCopies M) p roundIdemFp32Lhs s = some (fun _ => a) ∧
      evalOp (sourceCopies M) p roundIdemFp32Rhs s = some (fun _ => a) := by
  simp only [roundIdemFp32Lhs, roundIdemFp32Rhs, eval_fp32Copy,
    eval_loadedFp32, ha, and_self]

theorem eval_accWidenFp32_success {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (a b c : α)
    (ha : s.regs .fp32 [] "a" = some (fun _ => a))
    (hb : s.regs .fp32 [] "b" = some (fun _ => b))
    (hc : s.regs .fp32 [] "c" = some (fun _ => c)) :
    evalOp (sourceCopies M) p accWidenFp32Lhs s =
        some (fun _ => accWidenFp32Value M a b c) ∧
      evalOp (sourceCopies M) p accWidenFp32Rhs s =
        some (fun _ => accWidenFp32Value M a b c) := by
  rw [eval_accWidenFp32]
  simp only [accWidenFp32Rhs, eval_fp32Add, eval_loadedFp32, ha, hb, hc]
  exact ⟨rfl, rfl⟩

/-- The output expression already has the store's element dtype. -/
def scalarStore {dtype : TileDType} (out : Region dtype) (offset : Nat)
    (e : Op dtype []) : List ComputeStmt :=
  [.store dtype [] (.region out (.constNat offset)) (.alg e) .none]

theorem run_scalarStore {α : Type} [Inhabited α] (M : Algebra α)
    {dtype : TileDType} (out : Region dtype) (offset : Nat) (e : Op dtype [])
    (s : State α) (v : Structural.Value α dtype)
    (h : evalOp M none e s = some (fun _ => v)) :
    run M (scalarStore out offset e) s =
      some (s.write (Region.cast out) offset (.mk dtype v)) := by
  simp [scalarStore, run, step, evalExpr, h, store, evalOp_unfold,
    TileShape.allIndices]

/-- Both ROUND-IDEM/fp32 paths succeed and write the original fp32 value.
The final states are equal, so all other memory and registers are preserved. -/
theorem roundIdemFp32_store {α : Type} [Inhabited α] (M : Algebra α)
    (out : Region .fp32) (offset : Nat) (s : State α) (a : α)
    (ha : s.regs .fp32 [] "a" = some (fun _ => a)) :
    run (sourceCopies M) (scalarStore out offset roundIdemFp32Lhs) s =
        some (s.write (Region.cast out) offset (.mk .fp32 a)) ∧
      run (sourceCopies M) (scalarStore out offset roundIdemFp32Rhs) s =
        some (s.write (Region.cast out) offset (.mk .fp32 a)) := by
  obtain ⟨hl, hr⟩ := eval_roundIdemFp32_success M none s a ha
  exact ⟨run_scalarStore _ _ _ _ _ _ hl, run_scalarStore _ _ _ _ _ _ hr⟩

/-- Both ACC-WIDEN/fp32 paths succeed with exactly the same ordered binary
calls. No numerical assumption, commutativity or associativity is used. -/
theorem accWidenFp32_store {α : Type} [Inhabited α] (M : Algebra α)
    (out : Region .fp32) (offset : Nat) (s : State α) (a b c : α)
    (ha : s.regs .fp32 [] "a" = some (fun _ => a))
    (hb : s.regs .fp32 [] "b" = some (fun _ => b))
    (hc : s.regs .fp32 [] "c" = some (fun _ => c)) :
    run (sourceCopies M) (scalarStore out offset accWidenFp32Lhs) s =
        some (s.write (Region.cast out) offset (.mk .fp32 (accWidenFp32Value M a b c))) ∧
      run (sourceCopies M) (scalarStore out offset accWidenFp32Rhs) s =
        some (s.write (Region.cast out) offset (.mk .fp32 (accWidenFp32Value M a b c))) := by
  obtain ⟨hl, hr⟩ := eval_accWidenFp32_success M none s a b c ha hb hc
  exact ⟨run_scalarStore _ _ _ _ _ _ hl, run_scalarStore _ _ _ _ _ _ hr⟩

/- The two BF16-input profiles have identical CAST-MOVE graphs: this probe
uses `rnd(_, True)` regardless of LOW. Keep their final fp32-to-bf16 store
conversion explicit, including the common narrow/widen/narrow suffix. -/

def widenBf16 (e : Op .bf16 []) : Op .fp32 [] := .castFloat .bf16 .fp32 e
def narrowBf16 (e : Op .fp32 []) : Op .bf16 [] := .castFloat .fp32 .bf16 e
def rndBf16 (e : Op .fp32 []) : Op .fp32 [] := widenBf16 (narrowBf16 e)
def loadedBf16 (name : RegName) : Op .fp32 [] := widenBf16 (.ref .bf16 [] name)

/-- ROUND-IDEM and BF16-WIDEN-RETURN have this same AST in both BF16-input
profiles, including their common initial conversion of the loaded input. -/
def roundIdemBf16Lhs : Op .bf16 [] := narrowBf16 (rndBf16 (loadedBf16 "a"))
def roundIdemBf16Rhs : Op .bf16 [] := narrowBf16 (loadedBf16 "a")

def castMoveBf16Lhs : Op .bf16 [] :=
  narrowBf16 (rndBf16 (fp32Add (rndBf16 (loadedBf16 "a")) (rndBf16 (loadedBf16 "b"))))
def castMoveBf16Rhs : Op .bf16 [] :=
  narrowBf16 (rndBf16 (fp32Add (loadedBf16 "a") (loadedBf16 "b")))

/-- The common observable result; output conversions use the enclosing
context while the addition and its operands carry explicit fp32 precision. -/
def castMoveBf16Value {α : Type} (M : Algebra α) (p : Option ComputeDType)
    (a b : α) : α :=
  M.cast p .fp32 .bf16 (M.cast p .bf16 .fp32 (M.cast p .fp32 .bf16
    (M.binary (some .fp32) .fp32 .add
      (M.cast (some .fp32) .bf16 .fp32 a) (M.cast (some .fp32) .bf16 .fp32 b))))

theorem eval_widenBf16 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (e : Op .bf16 []) (s : State α) :
    evalOp (sourceCopies M) p (widenBf16 e) s =
      (evalOp (sourceCopies M) p e s).map (fun v i => M.cast p .bf16 .fp32 (v i)) := by
  rw [widenBf16, evalOp_unfold]
  change (evalOp (sourceCopies M) p e s).bind
    (fun v => some (fun i => M.cast p .bf16 .fp32 (v i))) = _
  cases evalOp (sourceCopies M) p e s <;> rfl

theorem eval_narrowBf16 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (e : Op .fp32 []) (s : State α) :
    evalOp (sourceCopies M) p (narrowBf16 e) s =
      (evalOp (sourceCopies M) p e s).map (fun v i => M.cast p .fp32 .bf16 (v i)) := by
  rw [narrowBf16, evalOp_unfold]
  change (evalOp (sourceCopies M) p e s).bind
    (fun v => some (fun i => M.cast p .fp32 .bf16 (v i))) = _
  cases evalOp (sourceCopies M) p e s <;> rfl

theorem eval_loadedBf16 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (name : RegName) (s : State α) :
    evalOp (sourceCopies M) p (loadedBf16 name) s =
      (s.regs .bf16 [] name).map (fun v i => M.cast p .bf16 .fp32 (v i)) := by
  rw [loadedBf16, eval_widenBf16, evalOp_unfold]

/-- Only the input's local cast round trip is a premise, not an equality of
the output expressions. The concrete-cast specialization below discharges it. -/
theorem eval_roundIdemBf16_success {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (a : α)
    (ha : s.regs .bf16 [] "a" = some (fun _ => a))
    (hcast : M.cast p .fp32 .bf16 (M.cast p .bf16 .fp32 a) = a) :
    evalOp (sourceCopies M) p roundIdemBf16Lhs s = some (fun _ => a) ∧
      evalOp (sourceCopies M) p roundIdemBf16Rhs s = some (fun _ => a) := by
  simp only [roundIdemBf16Lhs, roundIdemBf16Rhs, rndBf16,
    eval_narrowBf16, eval_widenBf16, eval_loadedBf16, ha, Option.map_some, hcast,
    and_self]

theorem eval_castMoveBf16_success {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (a b : α)
    (ha : s.regs .bf16 [] "a" = some (fun _ => a))
    (hb : s.regs .bf16 [] "b" = some (fun _ => b))
    (hcastA : M.cast (some .fp32) .fp32 .bf16 (M.cast (some .fp32) .bf16 .fp32 a) = a)
    (hcastB : M.cast (some .fp32) .fp32 .bf16 (M.cast (some .fp32) .bf16 .fp32 b) = b) :
    evalOp (sourceCopies M) p castMoveBf16Lhs s =
        some (fun _ => castMoveBf16Value M p a b) ∧
      evalOp (sourceCopies M) p castMoveBf16Rhs s =
        some (fun _ => castMoveBf16Value M p a b) := by
  simp only [castMoveBf16Lhs, castMoveBf16Rhs, rndBf16,
    eval_narrowBf16, eval_widenBf16, eval_fp32Add, eval_loadedBf16,
    ha, hb, Option.map_some, hcastA, hcastB]
  exact ⟨rfl, rfl⟩

theorem roundIdemBf16_store {α : Type} [Inhabited α] (M : Algebra α)
    (out : Region .bf16) (offset : Nat) (s : State α) (a : α)
    (ha : s.regs .bf16 [] "a" = some (fun _ => a))
    (hcast : M.cast none .fp32 .bf16 (M.cast none .bf16 .fp32 a) = a) :
    run (sourceCopies M) (scalarStore out offset roundIdemBf16Lhs) s =
        some (s.write (Region.cast out) offset (.mk .bf16 a)) ∧
      run (sourceCopies M) (scalarStore out offset roundIdemBf16Rhs) s =
        some (s.write (Region.cast out) offset (.mk .bf16 a)) := by
  obtain ⟨hl, hr⟩ := eval_roundIdemBf16_success M none s a ha hcast
  exact ⟨run_scalarStore _ _ _ _ _ _ hl, run_scalarStore _ _ _ _ _ _ hr⟩

theorem castMoveBf16_store {α : Type} [Inhabited α] (M : Algebra α)
    (out : Region .bf16) (offset : Nat) (s : State α) (a b : α)
    (ha : s.regs .bf16 [] "a" = some (fun _ => a))
    (hb : s.regs .bf16 [] "b" = some (fun _ => b))
    (hcastA : M.cast (some .fp32) .fp32 .bf16 (M.cast (some .fp32) .bf16 .fp32 a) = a)
    (hcastB : M.cast (some .fp32) .fp32 .bf16 (M.cast (some .fp32) .bf16 .fp32 b) = b) :
    run (sourceCopies M) (scalarStore out offset castMoveBf16Lhs) s =
        some (s.write (Region.cast out) offset (.mk .bf16 (castMoveBf16Value M none a b))) ∧
      run (sourceCopies M) (scalarStore out offset castMoveBf16Rhs) s =
        some (s.write (Region.cast out) offset (.mk .bf16 (castMoveBf16Value M none a b))) := by
  obtain ⟨hl, hr⟩ := eval_castMoveBf16_success M none s a b ha hb hcastA hcastB
  exact ⟨run_scalarStore _ _ _ _ _ _ hl, run_scalarStore _ _ _ _ _ _ hr⟩

/-- Distinct carriers retain actual BF16 and fp32 bit values. -/
abbrev CastValue := FP.Value .bf16 ⊕ FP.Value .fp32

/-- Bind just the two numerical casts to the shared concrete ieeeValue model.
Arithmetic remains the supplied opaque backend, so no arithmetic identity is
assumed. Other dtype channels and malformed carrier tags use `rest`. -/
def bf16FP32Casts (rest : Algebra CastValue) : Algebra CastValue :=
  { rest with cast := fun p src dst x => match src, dst, x with
      | .bf16, .fp32, .inl v => .inr (FP.cast .fp32 .ieeeValue v)
      | .fp32, .bf16, .inr v => .inl (FP.cast .bf16 .ieeeValue v)
      | _, _, _ => rest.cast p src dst x }

theorem bf16FP32Casts_return (rest : Algebra CastValue) (p : Option ComputeDType)
    (a : FP.Value .bf16) (ha : a.isNaN = false) :
    (bf16FP32Casts rest).cast p .fp32 .bf16
      ((bf16FP32Casts rest).cast p .bf16 .fp32 (.inl a)) = .inl a := by
  change Sum.inl (FP.cast .bf16 .ieeeValue (FP.cast .fp32 .ieeeValue a)) = Sum.inl a
  rw [CastExact.bf16_widen_return a ha]

/-- Strict ROUND-IDEM and BF16-WIDEN-RETURN specialization for both BF16-input
profiles. The non-NaN domain includes infinities, signed zero and subnormals. -/
theorem roundIdemBf16_store_concreteCasts [Inhabited CastValue] (rest : Algebra CastValue)
    (out : Region .bf16) (offset : Nat) (s : State CastValue) (a : FP.Value .bf16)
    (ha : s.regs .bf16 [] "a" = some (fun _ => Sum.inl a)) (hNaN : a.isNaN = false) :
    run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset roundIdemBf16Lhs) s =
        some (s.write (Region.cast out) offset (.mk .bf16 (Sum.inl a))) ∧
      run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset roundIdemBf16Rhs) s =
        some (s.write (Region.cast out) offset (.mk .bf16 (Sum.inl a))) :=
  roundIdemBf16_store _ out offset s (.inl a) ha (bf16FP32Casts_return rest none a hNaN)

/-- Strict CAST-MOVE specialization for bf16 and bf16_fp32 inputs. This uses
only the shared exact cast theorem; the ordered addition stays opaque. -/
theorem castMoveBf16_store_concreteCasts [Inhabited CastValue] (rest : Algebra CastValue)
    (out : Region .bf16) (offset : Nat) (s : State CastValue) (a b : FP.Value .bf16)
    (ha : s.regs .bf16 [] "a" = some (fun _ => Sum.inl a))
    (hb : s.regs .bf16 [] "b" = some (fun _ => Sum.inl b))
    (hNaN_a : a.isNaN = false) (hNaN_b : b.isNaN = false) :
    run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset castMoveBf16Lhs) s =
        some (s.write (Region.cast out) offset
          (.mk .bf16 (castMoveBf16Value (bf16FP32Casts rest) none (.inl a) (.inl b)))) ∧
      run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset castMoveBf16Rhs) s =
        some (s.write (Region.cast out) offset
          (.mk .bf16 (castMoveBf16Value (bf16FP32Casts rest) none (.inl a) (.inl b)))) :=
  castMoveBf16_store _ out offset s (.inl a) (.inl b) ha hb
    (bf16FP32Casts_return rest (some .fp32) a hNaN_a)
    (bf16FP32Casts_return rest (some .fp32) b hNaN_b)

#axiomsClean eval_roundIdemFp32
#axiomsClean eval_accWidenFp32
#axiomsClean roundIdemFp32_store
#axiomsClean accWidenFp32_store
#axiomsClean roundIdemBf16_store_concreteCasts
#axiomsClean castMoveBf16_store_concreteCasts

end VeriTile.Triton.FP.PrecisionIdentityLaws
