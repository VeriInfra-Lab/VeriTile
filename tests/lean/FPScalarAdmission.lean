import VeriTile.Triton.Float.ScalarArithmetic

namespace FPScalarAdmissionTests
open VeriTile.Triton.FP.ScalarArithmetic

/-- Historical reports must preserve the selected relation, precision and domain.
An accepted bf16 output cast cannot supply a bare fp32 law. A strict certificate
can make an atom available independently of report selection. -/
theorem report_matches (a : Atom) (h : a.report?.isSome = true) :
    (report a h).report.ruleID = a.ruleID ∧
    (report a h).report.input = "fp32" ∧ (report a h).report.compute = "fp32" ∧
    (report a h).report.accumulator = "fp32" ∧ (report a h).report.output = "fp32" ∧
    (report a h).guards = guards a := by
  cases a <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

example (a : Atom) : a.Available := by cases a <;> decide

end FPScalarAdmissionTests
