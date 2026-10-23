import VeriTile.Triton.Float.Countermodel
import VeriTile.Triton.Float.Integer

/-! A constructible interpretation of both the fixed strict software theory
and the selected scalar assumptions. Arithmetic uses actual FP32 words. The
operand domain contains +0, +1, and +2, on which all scalar candidates are exact.
Reduction, dot, mixed-format casts, and comparisons remain explicit abstract
operations. They provide an execution witness, not GPU conformance. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Number := FP.Value .fp32

def zero : Number := FP.Value.zero .fp32
def one : Number := FP.Value.ofBits .fp32 0x3f800000
def two : Number := FP.Value.ofBits .fp32 0x40000000
def sentinel : Number := FP.Value.ofBits .fp32 0x42c60000
def maskedScore : Number := FP.Value.ofBits .fp32 0xccbebc20

inductive Small where
  | zero | one | two
  deriving DecidableEq

def value : Small → Number
  | .zero => zero | .one => one | .two => two

/-- Every real literal reached by the execution witness keeps its value. -/
def literal (x : ℝ) : Number :=
  if x = 0 then zero else if x = 1 then one else if x = 2 then two
  else if x = 99 then sentinel else if x = -100000000 then maskedScore else zero

def rest : Algebra Number where
  literal := fun _ _ => literal
  negInf := FP.Value.infinity .fp32 Bool.true
  binary := fun _ _ op a b => match op with
    | .add => FP.add .ieeeValue a b | .sub => FP.sub .ieeeValue a b
    | .mul => FP.mul .ieeeValue a b | .div => FP.div .ieeeValue a b
    | .max => FP.MaximumExact.maximumValue a b | .pow => zero
  atomicAdd := fun _ => some (FP.add .ieeeValue)
  unary := fun _ _ _ => one
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareEq := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ x => x
  fromNat := fun _ n => FP.round .fp32 .ieeeValue (n : Rat)
  fromInt := fun _ n => FP.round .fp32 .ieeeValue (Rat.ofInt n)
  fp32Bits := fun b => ⟨b.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => zero
  reduceSum := fun _ {_} _ _ _ _ => one
  reduceSumTyped := fun _ d {_} _ _ _ => some (fun _ => ofFloat d one)
  scan := fun _ {_} _ _ _ v => some v
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => zero)
  dotAcc := fun _ _ _ out _ {_} {_} {_} {_} _ _ _ => some (fun _ => ofFloat out zero)
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out zero)

def model : Algebra Number := FP.Software.strictFP32 rest

def domain : FP.Guarded.Domain Number
  | .finite, x => ∃ a : Small, x = value a
  | .nonzero, x | .positive, x => x = one ∨ x = two

theorem domain_strict (kind : FP.GuardKind) (x : Number) (h : domain kind x) :
    FP.Guarded.strictDomain kind x := by
  cases kind with
  | finite =>
    obtain ⟨a, rfl⟩ := h
    change (value a).toRat?.isSome = Bool.true
    cases a <;> decide +kernel
  | nonzero =>
    rcases h with rfl | rfl
    · exact ⟨1, by decide +kernel, by decide⟩
    · exact ⟨2, by decide +kernel, by decide⟩
  | positive =>
    rcases h with rfl | rfl
    · exact ⟨1, by decide +kernel, by decide⟩
    · exact ⟨2, by decide +kernel, by decide⟩

#axiomsClean domain_strict
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel
