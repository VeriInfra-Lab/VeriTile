import VeriTile.Triton.Float.ScalarProbeExact
import VeriTile.Triton.Float.MaximumExact

/-! The twelve MAX probes use FP32 maximum after input widening, followed by
the actual output cast. The BF16-labelled source does not call `rnd` at any
maximum node. Finite guards include signed zero and subnormal inputs. -/
namespace VeriTile.Triton.FP.MaximumProbeExact
open ScalarProbeExact

inductive Rule where
  | commute | associate | idempotent | negativeInfinity
  deriving DecidableEq, Repr

def Rule.ruleID : Rule → String
  | .commute => "MAX-COMMUTE"
  | .associate => "MAX-ASSOC"
  | .idempotent => "MAX-IDEM"
  | .negativeInfinity => "MAX-NEG-INF"

def Rule.operands : Rule → List Operand
  | .commute => [.a, .b]
  | .associate => [.a, .b, .c]
  | .idempotent | .negativeInfinity => [.a]

def Rule.guards (r : Rule) : List OperandGuard :=
  r.operands.map (fun operand => ⟨operand.name, .finite⟩)

/-- Each expression has FP32 inputs and output; load and final store types
are supplied separately by the profile. There is deliberately no node cast. -/
inductive Expr where
  | ref32 (operand : Operand)
  | negativeInfinity32
  | maximum32 (a b : Expr)

def Expr.eval (input : Inputs .fp32) : Expr → Value .fp32
  | .ref32 operand => input.get operand
  | .negativeInfinity32 => Value.infinity .fp32 true
  | .maximum32 a b => MaximumExact.maximumValue (a.eval input) (b.eval input)

def lhs : Rule → Expr
  | .commute => .maximum32 (.ref32 .a) (.ref32 .b)
  | .associate => .maximum32 (.maximum32 (.ref32 .a) (.ref32 .b)) (.ref32 .c)
  | .idempotent => .maximum32 (.ref32 .a) (.ref32 .a)
  | .negativeInfinity => .maximum32 .negativeInfinity32 (.ref32 .a)

def rhs : Rule → Expr
  | .commute => .maximum32 (.ref32 .b) (.ref32 .a)
  | .associate => .maximum32 (.ref32 .a) (.maximum32 (.ref32 .b) (.ref32 .c))
  | .idempotent | .negativeInfinity => .ref32 .a

def execute (p : Profile) (e : Expr) (input : Inputs p.storageFormat) :
    Value p.storageFormat := store p (e.eval (loadInputs p input))

def Domain {f : Format} (r : Rule) (input : Inputs f) : Prop :=
  ∀ operand ∈ r.operands, (input.get operand).toRat?.isSome = true

def Certificate (p : Profile) (r : Rule) : Prop :=
  ∀ input : Inputs p.storageFormat, Domain r input →
    execute p (lhs r) input = execute p (rhs r) input

theorem all_certificates (p : Profile) (r : Rule) : Certificate p r := by
  intro input h
  have ha := load_finite p input.a (h .a (by cases r <;> simp [Rule.operands]))
  apply congrArg (store p)
  cases r with
  | commute =>
    have hb := load_finite p input.b (h .b (by simp [Rule.operands]))
    exact MaximumExact.maximumValue_commute _ _ ha hb
  | associate =>
    have hb := load_finite p input.b (h .b (by simp [Rule.operands]))
    have hc := load_finite p input.c (h .c (by simp [Rule.operands]))
    exact MaximumExact.maximumValue_assoc _ _ _ ha hb hc
  | idempotent => exact MaximumExact.maximumValue_idem _ ha
  | negativeInfinity => exact MaximumExact.maximumValue_neg_inf _ ha

#axiomsClean all_certificates

end VeriTile.Triton.FP.MaximumProbeExact
