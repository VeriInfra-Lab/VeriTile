import VeriTile.Triton.Float.IdentityExact
import VeriTile.Triton.Float.CastExact

/-! Typed source certificates for twelve historical scalar probe instances.

The source branches are `ADD-COMMUTE`/`MUL-COMMUTE` in
`experiments/floating_point/kernels.py` and `MUL-ONE`/`DIV-ONE` in
`experiments/floating_point/supplement/kernels.py`. In all three profiles the
common input prefix widens to FP32 and arithmetic operates in FP32. The BF16
profile's `rnd` converts to BF16 and back to FP32; the other profiles convert
to FP32. Each branch finally converts to the output storage format. In
particular, the identity candidate skips `rnd`, just as in the Python source.

These certificates contain proofs for fixed, format-indexed source terms.
They are software-model theorems, independent of experimental admission and
of GPU conformance. The finite domains retain signed zeros and subnormals.
-/
namespace VeriTile.Triton.FP.ScalarProbeExact

inductive Profile where
  | fp32 | bf16 | bf16Fp32
  deriving DecidableEq, Repr

def Profile.name : Profile → String
  | .fp32 => "fp32"
  | .bf16 => "bf16"
  | .bf16Fp32 => "bf16_fp32"

def Profile.storageFormat : Profile → Format
  | .fp32 => .fp32
  | .bf16 | .bf16Fp32 => .bf16

abbrev Profile.inputFormat (p : Profile) := p.storageFormat
abbrev Profile.outputFormat (p : Profile) := p.storageFormat

/-- The report's compute precision controls `rnd`; actual binary arithmetic
has already widened to FP32 in all three source instances. -/
def Profile.computeFormat : Profile → Format
  | .bf16 => .bf16
  | .fp32 | .bf16Fp32 => .fp32

def Profile.accumulatorFormat (_ : Profile) : Format := .fp32

/-- The common `.to(tl.float32)` following each typed input load. -/
def load (p : Profile) (x : Value p.storageFormat) : Value .fp32 :=
  FP.cast .fp32 .ieeeValue x

/-- The exact source `rnd` cast sequence, including its FP32 return type. -/
def rnd : Profile → Value .fp32 → Value .fp32
  | .bf16, x => FP.cast .fp32 .ieeeValue (FP.cast .bf16 .ieeeValue x)
  | .fp32, x | .bf16Fp32, x => FP.cast .fp32 .ieeeValue x

/-- The conversion performed by the final typed output store. -/
def store (p : Profile) (x : Value .fp32) : Value p.storageFormat :=
  FP.cast p.storageFormat .ieeeValue x

private theorem widen_finite (x : Value .bf16) (h : x.toRat?.isSome = true) :
    (FP.cast .fp32 .ieeeValue x).toRat?.isSome = true := by
  cases hd : x.decode with
  | finite s m =>
    rw [CastExact.cast_widen_of_decode_finite x s m hd]
    simp [Value.toRat?, CastExact.widenBits_decode_finite x s m hd]
  | infinity s => simp [Value.toRat?, hd] at h
  | nan s signaling payload => simp [Value.toRat?, hd] at h

theorem load_finite (p : Profile) (x : Value p.storageFormat)
    (h : x.toRat?.isSome = true) : (load p x).toRat?.isSome = true := by
  cases p with
  | fp32 =>
    exact (congrArg (fun value => value.toRat?.isSome)
      (CastExact.cast_same_of_finite x h)).trans h
  | bf16 => exact widen_finite x h
  | bf16Fp32 => exact widen_finite x h

private theorem finite_widen_return (x : Value .bf16) (h : x.toRat?.isSome = true) :
    FP.cast .bf16 .ieeeValue (FP.cast .fp32 .ieeeValue x) = x := by
  cases hd : x.decode with
  | finite s m => exact CastExact.bf16_widen_return_of_decode_finite x s m hd
  | infinity s => simp [Value.toRat?, hd] at h
  | nan s signaling payload => simp [Value.toRat?, hd] at h

/-- Final storage returns every finite input word, including both zero signs. -/
theorem store_load (p : Profile) (x : Value p.storageFormat)
    (h : x.toRat?.isSome = true) : store p (load p x) = x := by
  cases p with
  | fp32 =>
    exact (congrArg (FP.cast .fp32 .ieeeValue)
      (CastExact.cast_same_of_finite x h)).trans (CastExact.cast_same_of_finite x h)
  | bf16 => exact finite_widen_return x h
  | bf16Fp32 => exact finite_widen_return x h

/-- A finite BF16 input already lies in the range of the BF16 rounding step.
This is the required widening/return proof, not a BF16 arithmetic substitution. -/
theorem rnd_load (p : Profile) (x : Value p.storageFormat)
    (h : x.toRat?.isSome = true) : rnd p (load p x) = load p x := by
  cases p with
  | fp32 => exact CastExact.cast_same_of_finite _ (load_finite .fp32 x h)
  | bf16 =>
    exact congrArg (FP.cast .fp32 .ieeeValue) (finite_widen_return x h)
  | bf16Fp32 => exact CastExact.cast_same_of_finite _ (load_finite .bf16Fp32 x h)

inductive Operand where
  | a | b | c
  deriving DecidableEq, Repr

def Operand.name : Operand → String
  | .a => "a"
  | .b => "b"
  | .c => "c"

structure Inputs (f : Format) where
  a : Value f
  b : Value f
  c : Value f

def Inputs.get {f : Format} (input : Inputs f) : Operand → Value f
  | .a => input.a
  | .b => input.b
  | .c => input.c

/-- Both actual kernels load all three operands before selecting the branch. -/
def loadInputs (p : Profile) (input : Inputs p.storageFormat) : Inputs .fp32 :=
  ⟨load p input.a, load p input.b, load p input.c⟩

/-- Typed scalar source after the common load prefix. Binary constructors
explicitly operate in FP32; casts retain both source and destination formats. -/
inductive Expr : Format → Type where
  | ref32 (operand : Operand) : Expr .fp32
  | one32 : Expr .fp32
  | cast {src : Format} (dst : Format) (value : Expr src) : Expr dst
  | add32 (a b : Expr .fp32) : Expr .fp32
  | mul32 (a b : Expr .fp32) : Expr .fp32
  | div32 (a b : Expr .fp32) : Expr .fp32

def Expr.eval (input : Inputs .fp32) : {f : Format} → Expr f → Value f
  | _, .ref32 operand => input.get operand
  | _, .one32 => ScalarOpsLaws.one .fp32
  | _, .cast dst value => FP.cast dst .ieeeValue (value.eval input)
  | _, .add32 a b => FP.add .ieeeValue (a.eval input) (b.eval input)
  | _, .mul32 a b => FP.mul .ieeeValue (a.eval input) (b.eval input)
  | _, .div32 a b => FP.div .ieeeValue (a.eval input) (b.eval input)

def rndExpr : Profile → Expr .fp32 → Expr .fp32
  | .bf16, value => .cast .fp32 (.cast .bf16 value)
  | .fp32, value | .bf16Fp32, value => .cast .fp32 value

def storeExpr (p : Profile) (value : Expr .fp32) : Expr p.storageFormat :=
  .cast p.storageFormat value

@[simp] theorem eval_rndExpr (p : Profile) (input : Inputs .fp32) (value : Expr .fp32) :
    (rndExpr p value).eval input = rnd p (value.eval input) := by
  cases p <;> rfl

@[simp] theorem eval_storeExpr (p : Profile) (input : Inputs .fp32) (value : Expr .fp32) :
    (storeExpr p value).eval input = store p (value.eval input) := rfl

inductive Rule where
  | addCommute | mulCommute | mulOne | divOne
  deriving DecidableEq, Repr

def Rule.ruleID : Rule → String
  | .addCommute => "ADD-COMMUTE"
  | .mulCommute => "MUL-COMMUTE"
  | .mulOne => "MUL-ONE"
  | .divOne => "DIV-ONE"

def Rule.sourcePath : Rule → String
  | .addCommute | .mulCommute => "experiments/floating_point/kernels.py"
  | .mulOne | .divOne => "experiments/floating_point/supplement/kernels.py"

def Rule.operands : Rule → List Operand
  | .addCommute | .mulCommute => [.a, .b]
  | .mulOne | .divOne => [.a]

def Rule.guards (r : Rule) : List OperandGuard :=
  r.operands.map (fun operand => ⟨operand.name, .finite⟩)

/-- The original finite domain, with no condition on unused input channels. -/
def Domain {f : Format} (r : Rule) (input : Inputs f) : Prop :=
  ∀ operand ∈ r.operands, (input.get operand).toRat?.isSome = true

theorem domain_a {f : Format} (r : Rule) (input : Inputs f) (h : Domain r input) :
    input.a.toRat?.isSome = true := by
  apply h .a
  cases r <;> simp [Rule.operands]

/-- The fixed left source branch, including `rnd` and the final typed store. -/
def lhs (p : Profile) : Rule → Expr p.storageFormat
  | .addCommute => storeExpr p (rndExpr p (.add32 (.ref32 .a) (.ref32 .b)))
  | .mulCommute => storeExpr p (rndExpr p (.mul32 (.ref32 .a) (.ref32 .b)))
  | .mulOne => storeExpr p (rndExpr p (.mul32 (.ref32 .a) .one32))
  | .divOne => storeExpr p (rndExpr p (.div32 (.ref32 .a) .one32))

/-- Identity candidates have no intermediate `rnd`; the final store remains. -/
def rhs (p : Profile) : Rule → Expr p.storageFormat
  | .addCommute => storeExpr p (rndExpr p (.add32 (.ref32 .b) (.ref32 .a)))
  | .mulCommute => storeExpr p (rndExpr p (.mul32 (.ref32 .b) (.ref32 .a)))
  | .mulOne | .divOne => storeExpr p (.ref32 .a)

/-- Execute the actual common load/widen prefix and the typed scalar branch. -/
def execute (p : Profile) (source : Expr p.storageFormat) (input : Inputs p.storageFormat) :
    Value p.storageFormat := source.eval (loadInputs p input)

/-- A certificate fixes the profile, rule, source terms and finite domain.
It cannot be populated by an unrelated proposition or a theorem-name string. -/
def Certificate (p : Profile) (r : Rule) : Prop :=
  ∀ input : Inputs p.storageFormat, Domain r input →
    execute p (lhs p r) input = execute p (rhs p r) input

theorem add_commute (p : Profile) : Certificate p .addCommute := by
  intro input _
  simp only [execute, lhs, rhs, eval_storeExpr, eval_rndExpr, Expr.eval]
  rw [ScalarOpsLaws.add_comm]

theorem mul_commute (p : Profile) : Certificate p .mulCommute := by
  intro input _
  simp only [execute, lhs, rhs, eval_storeExpr, eval_rndExpr, Expr.eval]
  rw [ScalarOpsLaws.mul_comm]

theorem mul_one (p : Profile) : Certificate p .mulOne := by
  intro input h
  have hf := domain_a .mulOne input h
  simp only [execute, lhs, rhs, eval_storeExpr, eval_rndExpr, Expr.eval, loadInputs, Inputs.get]
  rw [IdentityExact.mul_one _ (load_finite p input.a hf), rnd_load p input.a hf]

theorem div_one (p : Profile) : Certificate p .divOne := by
  intro input h
  have hf := domain_a .divOne input h
  simp only [execute, lhs, rhs, eval_storeExpr, eval_rndExpr, Expr.eval, loadInputs, Inputs.get]
  rw [IdentityExact.div_one _ (load_finite p input.a hf), rnd_load p input.a hf]

/-- All twelve source-specific instances are backed by typed proofs. -/
theorem all_certificates (p : Profile) (r : Rule) : Certificate p r := by
  cases r with
  | addCommute => exact add_commute p
  | mulCommute => exact mul_commute p
  | mulOne => exact mul_one p
  | divOne => exact div_one p

/-- The source mask disables the store on inactive lanes. Its previous output
word is retained; active lanes use the entire source including output cast. -/
def executeMasked (p : Profile) (source : Expr p.storageFormat)
    (active : Bool) (input : Inputs p.storageFormat) (old : Value p.storageFormat) :
    Value p.storageFormat :=
  if active then execute p source input else old

theorem Certificate.masked {p : Profile} {r : Rule} (certificate : Certificate p r)
    (active : Bool) (input : Inputs p.storageFormat) (old : Value p.storageFormat)
    (h : active = true → Domain r input) :
    executeMasked p (lhs p r) active input old = executeMasked p (rhs p r) active input old := by
  cases active with
  | false => rfl
  | true => exact certificate input (h rfl)

#axiomsClean load_finite
#axiomsClean store_load
#axiomsClean rnd_load
#axiomsClean eval_rndExpr
#axiomsClean eval_storeExpr
#axiomsClean add_commute
#axiomsClean mul_commute
#axiomsClean mul_one
#axiomsClean div_one
#axiomsClean all_certificates
#axiomsClean Certificate.masked

end VeriTile.Triton.FP.ScalarProbeExact
