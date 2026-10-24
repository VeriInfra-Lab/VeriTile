import VeriTile.Triton.Float.ScalarArithmeticLaws
import VeriTile.Meta.StatementAudit

/-! Strict multiplication commutation for actual `.fp32` operands and output.
The fixed software theory certifies this typed execution directly; no coherence
between `.real` and `.fp32` channels and no new experimental binding is assumed.
The previous report binding remains available as historical metadata. -/
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

/-- Historical provenance for the previous experimental typed binding. This
entry is not part of the active assumptions or the strict derivation below. -/
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

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.scalar.assumptions
instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

theorem include_derivation {A B : Spec.Assumptions GuardedFragment}
    (subset : ∀ e ∈ A, e ∈ B) {left right : List GuardedFragment}
    (h : Spec.Derivation A left right) : Spec.Derivation B left right := by
  induction h with
  | refl code => exact .refl code
  | atom e he ha => exact .atom e (subset e he) ha
  | strict certificate => exact .strict certificate
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | frame pre post _ ih => exact .frame pre post ih

theorem scalar_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.scalar.assumptions M D := hM

/-- The original typed instructions, finite guards and FP32 default precision
are certified by executing them in the fixed software profile. The numeric
step reuses the shared all-word multiplication theorem. -/
theorem strict_certificate : StrictFragmentSound lhs rhs := by
  refine ⟨rfl, ?_⟩
  intro rest s hd
  obtain ⟨a, ha, _⟩ := hd ⟨"a", .finite⟩ (by simp [lhs, guards])
  obtain ⟨b, hb, _⟩ := hd ⟨"b", .finite⟩ (by simp [lhs, guards])
  change s.regs .fp32 [] "a" = some a at ha
  change s.regs .fp32 [] "b" = some b at hb
  simp [lhs, rhs, run, step, evalExpr, evalOp_unfold,
    numeric, ha, hb, scalarOutput, State.setReg, toFloat,
    fragmentAlgebra, Algebra.withDefaultPrecision, Software.strictFP32,
    Software.scalarFP32, Software.fp32Context, resolvePrecision]
  funext i
  exact ScalarOpsLaws.mul_comm .ieeeValue (a i) (b i)

theorem strict_admitted (R : Spec.Assumptions GuardedFragment) :
    Spec.Derivation R [lhs] [rhs] :=
  strict_derivation R strict_certificate

/-- Compatibility name: this derivation uses a strict certificate, with no
typed experimental entry or validation field. -/
theorem admitted (R : Rules) : Spec.Derivation R.assumptions [lhs] [rhs] :=
  strict_admitted R.assumptions

/-- The typed output is observed directly. No cast injectivity, real-channel
law, or desired whole-program equality is a premise. -/
theorem mul_comm_strict {α : Type} [Inhabited α] {R : Spec.Assumptions GuardedFragment}
    (M : Algebra α) (D : Domain α) (hM : Models R M D)
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
  have h := hM lhs rhs (strict_admitted R) t hd hd
  simp only [lhs, rhs, run, step, evalExpr, evalOp_unfold,
    numeric, State.setReg_same, t] at h
  simp [scalarOutput, State.setReg, toFloat] at h
  exact congrFun h PUnit.unit

/-- Compatibility wrapper for callers carrying scalar empirical rules. The
typed commutation itself uses only the fixed strict theory. -/
theorem mul_comm {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (a b : α) (ha : D .finite a) (hb : D .finite b) :
    M.binary (some .fp32) .fp32 .mul a b =
      M.binary (some .fp32) .fp32 .mul b a :=
  mul_comm_strict M D hM s a b ha hb

#axiomsClean strict_certificate
#axiomsClean strict_admitted
#axiomsClean admitted
#axiomsClean mul_comm_strict
#axiomsClean mul_comm
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.FP.TypedScalarArithmetic
