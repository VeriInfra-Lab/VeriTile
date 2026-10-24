import bench.tritonbench_g.context_attn_fwd.Proofs.FPModelData
import VeriTile.Triton.Float.ScalarArithmeticModel
import VeriTile.Triton.Float.TypedScalarArithmetic

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel
open VeriTile.Triton FP.Structural
noncomputable section

set_option maxHeartbeats 3200000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

/-- A task-local finite value table, evaluated using the shared FP operations.
No algebraic law for arbitrary FP32 inputs is inferred from this table. -/
theorem values_atom (atom : FP.ScalarArithmetic.Atom) (a b c : Number)
    (hg : FP.ScalarArithmetic.Inputs domain a b c atom) :
    FP.ScalarArithmetic.leftValue model a b c atom =
      FP.ScalarArithmetic.rightValue model a b c atom := by
  cases atom <;> simp only [FP.ScalarArithmetic.Inputs, domain] at hg
  case' addCommute | mulCommute | cancel =>
    rcases hg with ⟨⟨a, rfl⟩, ⟨b, rfl⟩⟩
    cases a <;> cases b
  case' addAssociate | mulAssociate | mulDistribute =>
    rcases hg with ⟨⟨a, rfl⟩, ⟨b, rfl⟩, ⟨c, rfl⟩⟩
    cases a <;> cases b <;> cases c
  case' addZero | mulOne | divOne =>
    rcases hg with ⟨a, rfl⟩
    cases a
  case' divMulRcp =>
    rcases hg with ⟨⟨a, rfl⟩, _, rfl | rfl⟩ <;> cases a
  case' mulRcpCancel =>
    rcases hg with ⟨_, rfl | rfl⟩
  all_goals
    simp only [FP.ScalarArithmetic.leftValue, FP.ScalarArithmetic.rightValue,
      FP.ScalarArithmetic.add, FP.ScalarArithmetic.sub, FP.ScalarArithmetic.mul,
      FP.ScalarArithmetic.div, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.one] <;>
    simp [model, FP.Software.strictFP32, FP.Software.scalarFP32,
      FP.Software.fp32Context, rest, literal] <;> decide +kernel

theorem arithmetic_atom (atom : FP.ScalarArithmetic.Atom) :
    FP.FragmentSound model domain (FP.ScalarArithmetic.lhs atom)
      (FP.ScalarArithmetic.rhs atom) :=
  FP.ScalarArithmetic.fragmentSound_of_values model domain atom (values_atom atom)

theorem model_satisfies_rules (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain := by
  apply FP.models_of_atoms
  · intro entry member
    obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp member
    unfold FP.ScalarArithmetic.Atom.entry? at he
    split at he
    · simp at he
    · cases hr : atom.report? with
      | none => simp [hr] at he
      | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (arithmetic_atom atom) .nil
  · exact FP.strict_software_interpretation rest domain domain_strict

theorem model_satisfies_typed_rules (R : FP.TypedScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain :=
  model_satisfies_rules R.scalar

#axiomsClean domain_strict
#axiomsClean values_atom
#axiomsClean model_satisfies_rules
#axiomsClean model_satisfies_typed_rules
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel
