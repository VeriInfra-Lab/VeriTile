import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.ScalarOpsLaws
import VeriTile.Triton.Float.CastExact.SameFormat

/-! Concrete FP32 identity fragments and their exact execution bridges.

`MUL-ONE` and `DIV-ONE` use the existing finite operand domain. The concrete
software configuration is explicit, and no GPU implementation claim follows.
`ADD-ZERO` has a signed-zero counterexample in that same finite domain, so its
existing numerical admission cannot be promoted to strict word equality.
-/
namespace VeriTile.Triton.FP.IdentityExact
open Structural Guarded

abbrev Word := FP.Value .fp32

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The usual finite/nonzero/positive interpretation, without excluding
negative zero or subnormal words from the finite guard. -/
def domain : Guarded.Domain Word
  | .finite, x => x.toRat?.isSome = true
  | .nonzero, x => ∃ q, x.toRat? = some q ∧ q ≠ 0
  | .positive, x => ∃ q, x.toRat? = some q ∧ 0 < q

theorem one_fp32 :
    ScalarOpsLaws.one .fp32 = FP.Value.ofBits .fp32 0x3f800000 := rfl

/-- Multiplication by positive one preserves every finite word with
nearest-even rounding and input/result flushing disabled. -/
theorem mul_one {f : Format} (x : FP.Value f) (finite : x.toRat?.isSome = true) :
    FP.mul .ieeeValue x (ScalarOpsLaws.one f) = x :=
  (ScalarOpsLaws.mul_one_cast .ieeeValue x).trans
    (CastExact.cast_same_of_finite x finite)

/-- Division by positive one has the same full finite-word domain. -/
theorem div_one {f : Format} (x : FP.Value f) (finite : x.toRat?.isSome = true) :
    FP.div .ieeeValue x (ScalarOpsLaws.one f) = x :=
  (ScalarOpsLaws.div_one_cast .ieeeValue x).trans
    (CastExact.cast_same_of_finite x finite)

theorem mul_one_fp32 (x : Word) (finite : domain .finite x) :
    FP.mul .ieeeValue x (FP.Value.ofBits .fp32 0x3f800000) = x := by
  simpa only [one_fp32] using mul_one x finite

theorem div_one_fp32 (x : Word) (finite : domain .finite x) :
    FP.div .ieeeValue x (FP.Value.ofBits .fp32 0x3f800000) = x := by
  simpa only [one_fp32] using div_one x finite

/-- Execute the actual `MUL-ONE` left fragment in every configuration.
Flushing and NaN canonicalization remain visible in the result cast. -/
theorem run_mulOne_lhs_cast (rest : Algebra Word) (cfg : Config)
    (s : State Word) (x : Word)
    (hx : s.regs .real [] "a" = some (fun _ => x)) :
    run (Software.scalarFP32 rest cfg) (ScalarArithmetic.lhs .mulOne).code s =
      some (s.setReg "out" .real [] (fun _ => FP.cast .fp32 cfg x)) := by
  simp only [ScalarArithmetic.lhs, ScalarArithmetic.lhsCode, ScalarArithmetic.fragment,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, ScalarArithmetic.ref,
    ScalarArithmetic.times, ComputeDType.eraseDType, numeric, hx]
  simp [Software.scalarFP32, Software.fp32Context, ← one_fp32]
  change s.setReg "out" .real [] (fun _ => FP.mul cfg x (ScalarOpsLaws.one .fp32)) = _
  rw [ScalarOpsLaws.mul_one_cast]

/-- Execute the actual `DIV-ONE` left fragment, retaining the configuration. -/
theorem run_divOne_lhs_cast (rest : Algebra Word) (cfg : Config)
    (s : State Word) (x : Word)
    (hx : s.regs .real [] "a" = some (fun _ => x)) :
    run (Software.scalarFP32 rest cfg) (ScalarArithmetic.lhs .divOne).code s =
      some (s.setReg "out" .real [] (fun _ => FP.cast .fp32 cfg x)) := by
  simp only [ScalarArithmetic.lhs, ScalarArithmetic.lhsCode, ScalarArithmetic.fragment,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, ScalarArithmetic.ref,
    ScalarArithmetic.divide, ComputeDType.eraseDType, numeric, hx]
  simp [Software.scalarFP32, Software.fp32Context, ← one_fp32]
  change s.setReg "out" .real [] (fun _ => FP.div cfg x (ScalarOpsLaws.one .fp32)) = _
  rw [ScalarOpsLaws.div_one_cast]

theorem run_addZero_lhs (rest : Algebra Word) (cfg : Config)
    (s : State Word) (x : Word)
    (hx : s.regs .real [] "a" = some (fun _ => x)) :
    run (Software.scalarFP32 rest cfg) (ScalarArithmetic.lhs .addZero).code s =
      some (s.setReg "out" .real [] (fun _ => FP.add cfg x (FP.Value.zero .fp32))) := by
  simp [ScalarArithmetic.lhs, ScalarArithmetic.lhsCode, ScalarArithmetic.fragment,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, ScalarArithmetic.ref,
    ScalarArithmetic.plus, ComputeDType.eraseDType, numeric, hx,
    Software.scalarFP32, Software.fp32Context]
  rfl

/-- Each existing identity right fragment reads `a` without numerical work. -/
theorem run_identity_rhs (rest : Algebra Word) (cfg : Config)
    (s : State Word) (x : Word) (atom : ScalarArithmetic.Atom)
    (hatom : atom = .addZero ∨ atom = .mulOne ∨ atom = .divOne)
    (hx : s.regs .real [] "a" = some (fun _ => x)) :
    run (Software.scalarFP32 rest cfg) (ScalarArithmetic.rhs atom).code s =
      some (s.setReg "out" .real [] (fun _ => x)) := by
  rcases hatom with rfl | rfl | rfl <;>
    simp [ScalarArithmetic.rhs, ScalarArithmetic.rhsCode, ScalarArithmetic.fragment,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ScalarArithmetic.ref,
      ComputeDType.eraseDType, hx]

theorem finite_guard (s : State Word) (x : Word) (atom : ScalarArithmetic.Atom)
    (hatom : atom = .addZero ∨ atom = .mulOne ∨ atom = .divOne)
    (hx : s.regs .real [] "a" = some (fun _ => x)) (finite : x.toRat?.isSome = true) :
    ScalarDomain domain (ScalarArithmetic.guards atom) s := by
  intro g hg
  rcases hatom with rfl | rfl | rfl <;>
    simp only [ScalarArithmetic.guards, List.mem_singleton] at hg <;>
    subst g <;> exact ⟨fun _ => x, hx, finite⟩

private theorem finite_input (s : State Word) (atom : ScalarArithmetic.Atom)
    (hatom : atom = .addZero ∨ atom = .mulOne ∨ atom = .divOne)
    (hg : ScalarDomain domain (ScalarArithmetic.lhs atom).guards s) :
    ∃ x : Word, s.regs .real [] "a" = some (fun _ => x) ∧ x.toRat?.isSome = true := by
  have member : (⟨"a", .finite⟩ : OperandGuard) ∈ (ScalarArithmetic.lhs atom).guards := by
    rcases hatom with rfl | rfl | rfl <;>
      simp [ScalarArithmetic.lhs, ScalarArithmetic.guards]
  rcases hg _ member with ⟨v, hv, hf⟩
  refine ⟨v PUnit.unit, ?_, hf⟩
  exact hv.trans (congrArg some (funext fun i => congrArg v (Subsingleton.elim i PUnit.unit)))

/-- The actual guarded `MUL-ONE` fragments both execute to the same full
state, using their existing precision, code, and finite operand guard. -/
theorem mulOne_execution (rest : Algebra Word) (s : State Word)
    (hg : ScalarDomain domain (ScalarArithmetic.lhs .mulOne).guards s) :
    ∃ t,
      run (fragmentAlgebra (ScalarArithmetic.lhs .mulOne)
        (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.lhs .mulOne).code s = some t ∧
      run (fragmentAlgebra (ScalarArithmetic.rhs .mulOne)
        (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.rhs .mulOne).code s = some t := by
  rcases finite_input s .mulOne (Or.inr (Or.inl rfl)) hg with ⟨x, hx, hf⟩
  refine ⟨s.setReg "out" .real [] (fun _ => x), ?_, ?_⟩
  · change run (Software.scalarFP32 rest .ieeeValue) (ScalarArithmetic.lhs .mulOne).code s = _
    simpa only [CastExact.cast_same_of_finite x hf] using
      run_mulOne_lhs_cast rest .ieeeValue s x hx
  · exact run_identity_rhs rest .ieeeValue s x .mulOne (Or.inr (Or.inl rfl)) hx

/-- The corresponding successful execution bridge for `DIV-ONE`. -/
theorem divOne_execution (rest : Algebra Word) (s : State Word)
    (hg : ScalarDomain domain (ScalarArithmetic.lhs .divOne).guards s) :
    ∃ t,
      run (fragmentAlgebra (ScalarArithmetic.lhs .divOne)
        (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.lhs .divOne).code s = some t ∧
      run (fragmentAlgebra (ScalarArithmetic.rhs .divOne)
        (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.rhs .divOne).code s = some t := by
  rcases finite_input s .divOne (Or.inr (Or.inr rfl)) hg with ⟨x, hx, hf⟩
  refine ⟨s.setReg "out" .real [] (fun _ => x), ?_, ?_⟩
  · change run (Software.scalarFP32 rest .ieeeValue) (ScalarArithmetic.lhs .divOne).code s = _
    simpa only [CastExact.cast_same_of_finite x hf] using
      run_divOne_lhs_cast rest .ieeeValue s x hx
  · exact run_identity_rhs rest .ieeeValue s x .divOne (Or.inr (Or.inr rfl)) hx

theorem mulOne_fragment_exact (rest : Algebra Word) (s : State Word)
    (hg : ScalarDomain domain (ScalarArithmetic.lhs .mulOne).guards s) :
    run (fragmentAlgebra (ScalarArithmetic.lhs .mulOne)
      (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.lhs .mulOne).code s =
    run (fragmentAlgebra (ScalarArithmetic.rhs .mulOne)
      (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.rhs .mulOne).code s := by
  rcases mulOne_execution rest s hg with ⟨_, hl, hr⟩
  exact hl.trans hr.symm

theorem divOne_fragment_exact (rest : Algebra Word) (s : State Word)
    (hg : ScalarDomain domain (ScalarArithmetic.lhs .divOne).guards s) :
    run (fragmentAlgebra (ScalarArithmetic.lhs .divOne)
      (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.lhs .divOne).code s =
    run (fragmentAlgebra (ScalarArithmetic.rhs .divOne)
      (Software.scalarFP32 rest .ieeeValue)) (ScalarArithmetic.rhs .divOne).code s := by
  rcases divOne_execution rest s hg with ⟨_, hl, hr⟩
  exact hl.trans hr.symm

theorem positive_zero_finite : domain .finite (FP.Value.zero .fp32) := by
  change (FP.Value.zero .fp32).toRat?.isSome = true
  decide +kernel

theorem negative_zero_finite : domain .finite (FP.Value.zero .fp32 true) := by
  change (FP.Value.zero .fp32 true).toRat?.isSome = true
  decide +kernel

theorem add_negative_zero_positive_zero :
    FP.add .ieeeValue (FP.Value.zero .fp32 true) (FP.Value.zero .fp32) =
      FP.Value.zero .fp32 := by
  decide +kernel

theorem zero_signs_distinct : FP.Value.zero .fp32 ≠ FP.Value.zero .fp32 true := by
  decide +kernel

/-- Both sides have rational value zero, although the words differ. This is
only the observation for this counterexample, not a whole-domain error bound. -/
theorem addZero_negative_zero_same_value :
    (FP.add .ieeeValue (FP.Value.zero .fp32 true) (FP.Value.zero .fp32)).toRat? =
      (FP.Value.zero .fp32 true).toRat? := by
  decide +kernel

/-- The existing finite-domain candidate fails as a strict scalar identity. -/
theorem addZero_not_exact_on_finite :
    ¬ (∀ x : Word, domain .finite x → FP.add .ieeeValue x (FP.Value.zero .fp32) = x) := by
  intro h
  exact zero_signs_distinct ((add_negative_zero_positive_zero.symm).trans
    (h _ negative_zero_finite))

/-- The counterexample executes the actual registered fragments successfully,
with both unchanged finite guards, and yields different output words. -/
theorem addZero_execution_counterexample (rest : Algebra Word) (s : State Word) :
    let input := s.setReg "a" .real [] (fun _ => FP.Value.zero .fp32 true)
    ScalarDomain domain (ScalarArithmetic.lhs .addZero).guards input ∧
    ScalarDomain domain (ScalarArithmetic.rhs .addZero).guards input ∧
    ∃ left right,
      run (Software.scalarFP32 rest .ieeeValue) (ScalarArithmetic.lhs .addZero).code
        input = some left ∧
      run (Software.scalarFP32 rest .ieeeValue) (ScalarArithmetic.rhs .addZero).code
        input = some right ∧
      scalarOutput (ScalarArithmetic.lhs .addZero).dtype left ≠
        scalarOutput (ScalarArithmetic.rhs .addZero).dtype right := by
  dsimp only
  let input := s.setReg "a" .real [] (fun _ => FP.Value.zero .fp32 true)
  have hx : input.regs .real [] "a" = some (fun _ => FP.Value.zero .fp32 true) := by
    simp [input]
  have hg := finite_guard input _ .addZero (Or.inl rfl) hx negative_zero_finite
  refine ⟨hg, hg, input.setReg "out" .real [] (fun _ => FP.Value.zero .fp32),
    input.setReg "out" .real [] (fun _ => FP.Value.zero .fp32 true), ?_,
    run_identity_rhs rest .ieeeValue input _ .addZero (Or.inl rfl) hx, ?_⟩
  · simpa only [add_negative_zero_positive_zero] using
      run_addZero_lhs rest .ieeeValue input _ hx
  · simp only [ScalarArithmetic.lhs, ScalarArithmetic.rhs, scalarOutput_real,
      State.setReg_same, ne_eq, Option.some.injEq]
    intro h
    exact zero_signs_distinct (congrFun h PUnit.unit)

#axiomsClean one_fp32
#axiomsClean mul_one
#axiomsClean div_one
#axiomsClean mul_one_fp32
#axiomsClean div_one_fp32
#axiomsClean run_mulOne_lhs_cast
#axiomsClean run_divOne_lhs_cast
#axiomsClean run_addZero_lhs
#axiomsClean run_identity_rhs
#axiomsClean finite_guard
#axiomsClean mulOne_execution
#axiomsClean divOne_execution
#axiomsClean mulOne_fragment_exact
#axiomsClean divOne_fragment_exact
#axiomsClean positive_zero_finite
#axiomsClean negative_zero_finite
#axiomsClean add_negative_zero_positive_zero
#axiomsClean zero_signs_distinct
#axiomsClean addZero_negative_zero_same_value
#axiomsClean addZero_not_exact_on_finite
#axiomsClean addZero_execution_counterexample

end VeriTile.Triton.FP.IdentityExact
