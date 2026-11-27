import VeriTile.Triton.Float.WordBackend
import VeriTile.Triton.Float.ScalarArithmeticModel

/-!
The scalar arithmetic rule table has a rounded IEEE model on a small domain.

`Guarded.Models R M D` is a premise of every `≡[R]` conclusion. If no backend
satisfied it, such a conclusion would hold for any two programs. This file
exhibits one: every strict FP32 backend over one-word registers, hence
`mixedFP32`, validates all eleven scalar arithmetic candidates when the finite
operands are `+0` and `+1` and the nonzero and positive operands are `+1`.
Sums and products of these operands are exact, so the rounded operations
agree with the relaxed rules there.

The domain is deliberately small. It shows that the premise is satisfiable by
real rounding; it does not claim that the rules hold on a larger domain, and
`Counterexamples.lean` shows that several of them fail on all finite words.
-/
namespace VeriTile.Triton.FP.Software.UnitOperands
open Structural Guarded ScalarArithmetic

/-- The two operands of the small domain: `+0` and `+1`. -/
def word (b : Bool) : FP.Value .fp32 :=
  if b then FP.Value.ofBits .fp32 0x3f800000 else FP.Value.zero .fp32

/-- Finite rule operands are `+0` and `+1`; division also needs `+1`. -/
def domain : Domain (FP.Value .fp32)
  | .finite => fun x => ∃ b : Bool, x = word b
  | .positive | .nonzero => fun x => x = word Bool.true

/-- The small domain lies inside the operand domain of the strict certificates. -/
theorem domain_strict (kind : GuardKind) (value : FP.Value .fp32)
    (member : domain kind value) : strictDomain kind value := by
  cases kind
  · obtain ⟨b, rfl⟩ := member
    cases b
    · change (FP.Value.zero .fp32).toRat?.isSome = Bool.true
      decide +kernel
    · change (FP.Value.ofBits .fp32 0x3f800000).toRat?.isSome = Bool.true
      decide +kernel
  · subst member
    exact ⟨1, by decide +kernel, by decide⟩
  · subst member
    exact ⟨1, by decide +kernel, by decide⟩

/-- Every arithmetic candidate is an exact guarded fragment on this domain. -/
theorem arithmetic_atom (rest : Algebra (FP.Value .fp32)) (atom : Atom) :
    FragmentSound (strictFP32 rest) domain (lhs atom) (rhs atom) := by
  apply fragmentSound_of_values
  intro a b c h
  have addWord (a b : FP.Value .fp32) :
      ScalarArithmetic.add (strictFP32 rest) a b = FP.add .ieeeValue a b := by
    simp [ScalarArithmetic.add, strictFP32, scalarFP32, fp32Context]
  have subWord (a b : FP.Value .fp32) :
      ScalarArithmetic.sub (strictFP32 rest) a b = FP.sub .ieeeValue a b := by
    simp [ScalarArithmetic.sub, strictFP32, scalarFP32, fp32Context]
  have mulWord (a b : FP.Value .fp32) :
      ScalarArithmetic.mul (strictFP32 rest) a b = FP.mul .ieeeValue a b := by
    simp [ScalarArithmetic.mul, strictFP32, scalarFP32, fp32Context]
  have divWord (a b : FP.Value .fp32) :
      ScalarArithmetic.div (strictFP32 rest) a b = FP.div .ieeeValue a b := by
    simp [ScalarArithmetic.div, strictFP32, scalarFP32, fp32Context]
  have zeroWord : ScalarArithmetic.zero (strictFP32 rest) = word Bool.false := by
    simp [ScalarArithmetic.zero, word, strictFP32, scalarFP32, fp32Context]
  have oneWord : ScalarArithmetic.one (strictFP32 rest) = word Bool.true := by
    simp [ScalarArithmetic.one, word, strictFP32, scalarFP32, fp32Context]
  cases atom <;> simp only [Inputs, domain] at h
  case' addCommute | mulCommute | cancel =>
    rcases h with ⟨⟨i, rfl⟩, ⟨j, rfl⟩⟩
    cases i <;> cases j
  case' addAssociate | mulAssociate | mulDistribute =>
    rcases h with ⟨⟨i, rfl⟩, ⟨j, rfl⟩, ⟨k, rfl⟩⟩
    cases i <;> cases j <;> cases k
  case' addZero | mulOne | divOne =>
    obtain ⟨i, rfl⟩ := h
    cases i
  case' divMulRcp =>
    rcases h with ⟨⟨i, rfl⟩, _, rfl⟩
    cases i
  case' mulRcpCancel =>
    obtain ⟨_, rfl⟩ := h
  all_goals simp only [leftValue, rightValue, addWord, subWord, mulWord, divWord, zeroWord, oneWord]
  all_goals decide +kernel

/-- Each admitted entry of the arithmetic table holds on the small domain. -/
theorem admitted_entries (rest : Algebra (FP.Value .fp32)) (R : Rules) (entry)
    (member : entry ∈ R.assumptions) :
    List.Forall₂ (FragmentSound (strictFP32 rest) domain) entry.rule.lhs entry.rule.rhs := by
  obtain ⟨atom, _, found⟩ := List.mem_filterMap.mp member
  unfold Atom.entry? at found
  split at found
  · cases found
  cases report : atom.report? with
  | none => simp [report] at found
  | some row =>
    simp only [report, Option.map_some, Option.some.injEq] at found
    subst entry
    exact .cons (arithmetic_atom rest atom) .nil

/-- A strict FP32 backend models the whole arithmetic rule table on the small
domain. The premise of an arithmetic `≡[R]` conclusion is therefore
satisfiable by rounded IEEE arithmetic. -/
theorem strict_models (rest : Algebra (FP.Value .fp32)) (R : Rules) :
    Models R.assumptions (strictFP32 rest) domain :=
  models_of_atoms _ domain R.assumptions (admitted_entries rest R)
    (fun lhs rhs certificate =>
      strict_software_interpretation rest domain domain_strict lhs rhs certificate)

/-- The mixed BF16/FP32 backend is one such model. -/
theorem mixed_models (rest : Algebra (FP.Value .fp32)) (R : Rules) :
    Models R.assumptions (mixedFP32 rest) domain :=
  strict_models (bf16Words rest) R

#axiomsClean domain_strict
#axiomsClean arithmetic_atom
#axiomsClean admitted_entries
#axiomsClean strict_models
#axiomsClean mixed_models
end VeriTile.Triton.FP.Software.UnitOperands
