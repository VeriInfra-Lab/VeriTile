import VeriTile.Triton.Float.ScalarArithmeticLaws
import VeriTile.Meta.StatementAudit

/-! A separate binding of the fp32 MUL-COMMUTE report to actual `.fp32`
operands and output. This does not enlarge `ScalarArithmetic.Rules` or infer
coherence between the `.real` and `.fp32` channels. Validation of this exact
binding is an additional, explicit premise. -/
namespace VeriTile.Triton.FP.TypedScalarArithmetic
open Structural Guarded
open scoped VeriTile.Spec

def guards : List OperandGuard := [⟨"a", .finite⟩, ⟨"b", .finite⟩]
def lhs : GuardedFragment := ⟨guards,
  [.assign .fp32 [] "out" (.alg
    (.mul .fp32 .nil (.ref .fp32 [] "a") (.ref .fp32 [] "b")))], .fp32, some .fp32⟩
def rhs : GuardedFragment := ⟨guards,
  [.assign .fp32 [] "out" (.alg
    (.mul .fp32 .nil (.ref .fp32 [] "b") (.ref .fp32 [] "a")))], .fp32, some .fp32⟩

/-- Provenance stays tied to the published fp32 row. The new binding has its
own identity and explicitly names both the Lean channel and finite input domain. -/
def report : ReportedRule :=
  { ReportedAdmission.fp32_mul_commute with
    key := ReportedAdmission.fp32_mul_commute.key ++ ":typed-fp32"
    configuration := Lean.Json.mkObj [
      ("source_report", ReportedAdmission.fp32_mul_commute.configuration),
      ("lean_binding", .str "typed-fp32 scalar a*b -> b*a; finite actual fp32 a,b; fp32 out"),
      ("binding_evidence", .str "bench/review-checkpoints/2026-10-14/typed-fp32-mul-binding.md")]
    description := "Typed fp32 MUL-COMMUTE; finite fp32 operands; separately validated binding of the published fp32 row" }

def entry : Spec.RuleEntry GuardedFragment := report.bind [lhs] [rhs]

structure Rules where
  scalar : ScalarArithmetic.Rules
  /-- Validation is for this new exact typed fragment, not the old real-channel atom. -/
  validatedTyped : Spec.EvidenceValidated entry.rule entry.evidence

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.scalar.assumptions ++ [entry]
instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

theorem include_derivation {A B : Spec.Assumptions GuardedFragment}
    (subset : ∀ e ∈ A, e ∈ B) {left right : List GuardedFragment}
    (h : Spec.Derivation A left right) : Spec.Derivation B left right := by
  induction h with
  | refl code => exact .refl code
  | atom e he ha => exact .atom e (subset e he) ha
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | frame pre post _ ih => exact .frame pre post ih

theorem scalar_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.scalar.assumptions M D :=
  fun left right h => hM left right
    (include_derivation (fun _ he => List.mem_append_left _ he) h)

theorem admitted (R : Rules) : Spec.Derivation R.assumptions [lhs] [rhs] := by
  apply Spec.Derivation.atom entry
  · exact List.mem_append_right _ (List.mem_cons_self ..)
  · exact report.admit _ _ R.validatedTyped

/-- The typed output is observed directly. No cast injectivity, real-channel
law, or desired whole-program equality is a premise. -/
theorem mul_comm {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (a b : α) (ha : D .finite a) (hb : D .finite b) :
    M.binary (some .fp32) .fp32 .mul a b =
      M.binary (some .fp32) .fp32 .mul b a := by
  let t := (s.setReg "a" .fp32 [] (fun _ => a)).setReg "b" .fp32 [] (fun _ => b)
  have hd : TypedScalarDomain D .fp32 guards t := by
    intro g hg
    simp only [guards, List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl
    · exact ⟨fun _ => a, by simp [t, State.setReg], ha⟩
    · exact ⟨fun _ => b, by simp [t, State.setReg], hb⟩
  have h := hM lhs rhs (admitted R) t hd hd
  simp only [lhs, rhs, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    ComputeDType.eraseDType, numeric, State.setReg_same, t] at h
  simp [scalarOutput, State.setReg, toFloat, bop] at h
  exact congrFun h PUnit.unit

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.FP.TypedScalarArithmetic
