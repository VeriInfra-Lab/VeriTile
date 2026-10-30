import VeriTile.Triton.Float.Countermodel
import VeriTile.Triton.Float.ScalarArithmetic

/-! A shared structural bridge for constructing interpretations of the scalar
rules. Numerical validity is supplied separately for the displayed domain. -/
namespace VeriTile.Triton.FP.ScalarArithmetic
open Structural Guarded

private theorem guarded_register {α : Type} (D : Domain α) (s : State α)
    (gs : List OperandGuard) (hd : ScalarDomain D gs s) (name : RegName)
    (hm : (⟨name, .finite⟩ : OperandGuard) ∈ gs) :
    ∃ a : α, s.regs .real [] name = some (fun _ => a) ∧ D .finite a := by
  obtain ⟨v, hv, hf⟩ := hd _ hm
  refine ⟨v PUnit.unit, hv.trans (congrArg some ?_), hf⟩
  funext i
  cases i
  rfl

private theorem guarded_nonzero {α : Type} (D : Domain α) (s : State α)
    (gs : List OperandGuard) (hd : ScalarDomain D gs s) (name : RegName)
    (hm : (⟨name, .nonzero⟩ : OperandGuard) ∈ gs) (a : α)
    (ha : s.regs .real [] name = some (fun _ => a)) : D .nonzero a := by
  obtain ⟨v, hv, hn⟩ := hd _ hm
  have hv' := Option.some.inj (hv.symm.trans ha)
  simpa only [hv'] using hn

set_option maxHeartbeats 1600000 in
/-- Domain-restricted scalar equalities interpret the actual registered
fragments, including their FP32 tags and operand guards. -/
theorem fragmentSound_of_values {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (atom : Atom)
    (law : ∀ a b c, Inputs D a b c atom →
      leftValue M a b c atom = rightValue M a b c atom) :
    FragmentSound M D (lhs atom) (rhs atom) := by
  refine ⟨rfl, fun s hd => ?_⟩
  change ScalarDomain D (guards atom) s at hd
  cases atom
  all_goals
    obtain ⟨a, ha, hfa⟩ := guarded_register D s _ hd "a" (by decide)
    try
      have h : ∃ b : α, s.regs .real [] "b" = some (fun _ => b) ∧ D .finite b := by
        apply guarded_register D s _ hd
        decide
      obtain ⟨b, hb, hfb⟩ := h
    try
      have h : ∃ c : α, s.regs .real [] "c" = some (fun _ => c) ∧ D .finite c := by
        apply guarded_register D s _ hd
        decide
      obtain ⟨c, hc, hfc⟩ := h
    try have hn := guarded_nonzero D s _ hd "b" (by decide) b hb
    try have hn := guarded_nonzero D s _ hd "a" (by decide) a ha
    first
      | have hv := law a b c ⟨hfa,hfb,hfc⟩
      | have hv := law a b a ⟨hfa,hfb,hn⟩
      | have hv := law a b a ⟨hfa,hfb⟩
      | have hv := law a a a ⟨hfa,hn⟩
      | have hv := law a a a hfa
    clear law
    simp only [leftValue, rightValue, add, sub, mul, div, zero, one] at hv
    simp only [lhs, rhs, lhsCode, rhsCode, fragment, fragmentAlgebra,
      ref, plus, minus, times, divide, Reciprocal.lhs, Reciprocal.rhs,
      Reciprocal.div, Reciprocal.inv, Reciprocal.mul, Reciprocal.ref,
      run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, ha]
    simp_all [State.setReg]
    funext i
    exact hv

#axiomsClean fragmentSound_of_values

end VeriTile.Triton.FP.ScalarArithmetic
