import VeriTile.Triton.Float.ScalarArithmeticModel
import bench.examples.StableLogSumExp.Proofs.FP

/-! A constructible model of the strict software theory and the selected
experimental rules. Scalar arithmetic uses the actual FP32 software operations.
The finite operand domain contains +0 and +1; nonzero and positive operands
must equal +1. Every selected arithmetic rule is exact on these inputs.
The exponential residual algorithms return +1 and both log backends return
+0; comparisons choose the fallbacks. These unsupported primitives remain
explicit interpretations, so this is a logical countermodel, not a GPU
measurement or a claim about a complete transcendental implementation. -/
noncomputable section
namespace VeriTile.Bench.Examples.StableLogSumExpUnconditional.Model
open VeriTile Triton FP FP.Structural FP.Guarded

/-- The carrier retains all FP32 words; only the operand guard is restricted. -/
abbrev Number := FP.Value .fp32

def zero : Number := FP.Value.zero .fp32
def one : Number := FP.Value.ofBits .fp32 0x3f800000
def value (b : Bool) : Number := if b then one else zero

def rest : Algebra Number where
  literal := fun _ _ r => if r = 0 then zero else one
  negInf := FP.Value.infinity .fp32 Bool.true
  binary := fun _ _ _ _ _ => zero
  unary := fun _ op _ => match op with
    | .exp | .libdeviceExp => one
    | _ => zero
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ => id
  fromNat := fun _ n => FP.round .fp32 .ieeeValue (n : Rat)
  fromInt := fun _ n => FP.round .fp32 .ieeeValue (Rat.ofInt n)
  fp32Bits := fun b => ⟨b.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => one
  reduceSum := fun _ {_} _ _ _ _ => zero

def algebra : Algebra Number := Software.strictFP32 rest

private theorem model_add (a b : Number) :
    algebra.binary (some .fp32) .real .add a b = FP.add .ieeeValue a b := rfl
private theorem model_sub (a b : Number) :
    algebra.binary (some .fp32) .real .sub a b = FP.sub .ieeeValue a b := rfl
private theorem model_mul (a b : Number) :
    algebra.binary (some .fp32) .real .mul a b = FP.mul .ieeeValue a b := rfl
private theorem model_div (a b : Number) :
    algebra.binary (some .fp32) .real .div a b = FP.div .ieeeValue a b := rfl
private theorem model_zero : algebra.literal (some .fp32) .real 0 = zero := by
  simp [algebra, Software.strictFP32, Software.scalarFP32, Software.fp32Context, zero]
private theorem model_one : algebra.literal (some .fp32) .real 1 = one := by
  simp [algebra, Software.strictFP32, Software.scalarFP32, Software.fp32Context, one]
private theorem model_tl_exp (x : Number) : algebra.unary (some .fp32) .exp x =
    ExponentialExact.tlExp (Software.strictTLExpResidual rest) x := rfl
private theorem model_libdevice_exp (x : Number) :
    algebra.unary (some .fp32) .libdeviceExp x =
      ExponentialExact.libdeviceExp (Software.strictLibdeviceExpResidual rest) x := rfl
private theorem model_log (x : Number) : algebra.unary (some .fp32) .log x = zero := rfl
private theorem model_libdevice_log (x : Number) :
    algebra.unary (some .fp32) .libdeviceLog x = zero := rfl
private theorem model_lt (p d) : algebra.compareLt p d = some (fun _ _ => Bool.false) := rfl
private theorem model_le (p d) : algebra.compareLe p d = some (fun _ _ => Bool.true) := rfl
private theorem model_neg_inf : algebra.negInf = FP.Value.infinity .fp32 Bool.true := rfl

def domain : Domain Number
  | .finite => fun a => ∃ b : Bool, a = value b
  | .nonzero | .positive => fun a => a = one

theorem domain_strict (kind : GuardKind) (x : Number) (h : domain kind x) :
    strictDomain kind x := by
  cases kind with
  | finite =>
    obtain ⟨b, rfl⟩ := h
    change (value b).toRat?.isSome = Bool.true
    cases b <;> decide +kernel
  | nonzero =>
    change x = one at h
    subst x
    exact ⟨1, by decide +kernel, by decide⟩
  | positive =>
    change x = one at h
    subst x
    exact ⟨1, by decide +kernel, by decide⟩

/-- Strict certificates use actual software execution and the proved guard
inclusion, without an extra interpretation hypothesis. -/
theorem strict_certificates (lhs rhs : List GuardedFragment)
    (h : fp32Theory.Certified lhs rhs) :
    List.Forall₂ (FragmentSound algebra domain) lhs rhs :=
  strict_software_interpretation rest domain domain_strict lhs rhs h

private theorem finite_register (s : State Number) (gs : List OperandGuard)
    (hd : ScalarDomain domain gs s) (name : RegName) (hm : ⟨name, .finite⟩ ∈ gs) :
    ∃ b : Bool, s.regs .real [] name = some (fun _ => value b) := by
  obtain ⟨v, hv, hf⟩ := hd _ hm
  obtain ⟨b, hb⟩ := hf
  refine ⟨b, hv.trans (congrArg some ?_)⟩
  funext i
  cases i
  exact hb

private theorem nonzero_register (s : State Number) (gs : List OperandGuard)
    (hd : ScalarDomain domain gs s) (name : RegName) (hm : ⟨name, .nonzero⟩ ∈ gs)
    (b : Bool) (hb : s.regs .real [] name = some (fun _ => value b)) : b = Bool.true := by
  obtain ⟨v, hv, hn⟩ := hd _ hm
  have eqv := Option.some.inj (hv.symm.trans hb)
  have hvb : value b = one := by simpa only [domain, eqv] using hn
  have hb' : value b = one ↔ b = Bool.true := by cases b <;> decide +kernel
  exact hb'.mp hvb

set_option linter.unusedSimpArgs false

set_option maxHeartbeats 1600000 in
theorem arithmetic_values (atom : ScalarArithmetic.Atom) (a b c : Number)
    (h : ScalarArithmetic.Inputs domain a b c atom) :
    ScalarArithmetic.leftValue algebra a b c atom =
      ScalarArithmetic.rightValue algebra a b c atom := by
  cases atom <;> simp only [ScalarArithmetic.Inputs] at h
  case' addCommute | mulCommute | cancel =>
    rcases h with ⟨⟨ia, rfl⟩, ⟨ib, rfl⟩⟩
    cases ia <;> cases ib
  case' addAssociate | mulAssociate | mulDistribute =>
    rcases h with ⟨⟨ia, rfl⟩, ⟨ib, rfl⟩, ⟨ic, rfl⟩⟩
    cases ia <;> cases ib <;> cases ic
  case' addZero | mulOne | divOne =>
    obtain ⟨ia, rfl⟩ := h
    cases ia
  case' divMulRcp =>
    rcases h with ⟨⟨ia, rfl⟩, _, hb⟩
    change b = one at hb
    subst b
    cases ia
  case' mulRcpCancel =>
    obtain ⟨_, ha⟩ := h
    change a = one at ha
    subst a
  all_goals simp only [ScalarArithmetic.leftValue, ScalarArithmetic.rightValue,
      ScalarArithmetic.add, ScalarArithmetic.sub, ScalarArithmetic.mul,
      ScalarArithmetic.div, ScalarArithmetic.zero, ScalarArithmetic.one,
      model_add, model_sub, model_mul, model_div, model_zero, model_one]
  all_goals decide +kernel

/-- The shared execution bridge keeps the numerical finite table separate
from FP32 fragment evaluation and its register/domain obligations. -/
theorem arithmetic_atom (atom : ScalarArithmetic.Atom) :
    FragmentSound algebra domain (ScalarArithmetic.lhs atom) (ScalarArithmetic.rhs atom) :=
  ScalarArithmetic.fragmentSound_of_values algebra domain atom (arithmetic_values atom)

set_option maxHeartbeats 1600000 in
theorem exponential_atom (atom : Exponential.Atom) :
    FragmentSound algebra domain atom.lhs atom.rhs := by
  refine ⟨rfl, fun s hd => ?_⟩
  change ScalarDomain domain atom.guards s at hd
  rcases atom with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _⟩
  all_goals
    try
      have h : ∃ a : Bool, s.regs .real [] "a" = some (fun _ => value a) := by
        apply finite_register s _ hd
        decide
      obtain ⟨a, ha⟩ := h
    try
      have h : ∃ b : Bool, s.regs .real [] "b" = some (fun _ => value b) := by
        apply finite_register s _ hd
        decide
      obtain ⟨b, hb⟩ := h
    simp only [Exponential.Atom.lhs, Exponential.Atom.rhs, Exponential.Backend.exp,
      ScalarArithmetic.fragment, ScalarArithmetic.ref, ScalarArithmetic.minus,
      ScalarArithmetic.divide, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, State.setReg_same]
    simp_all [State.setReg, model_sub, model_div, model_zero, model_one,
      model_tl_exp, model_libdevice_exp, model_neg_inf]
    all_goals try cases a
    all_goals try cases b
    all_goals funext i
    all_goals cases i
    all_goals decide +kernel

set_option maxHeartbeats 1600000 in
private theorem log_split (backend : LogExp.Backend) :
    FragmentSound algebra domain (LogExp.Atom.log_mul_split backend).lhs
      (LogExp.Atom.log_mul_split backend).rhs := by
  refine ⟨rfl, fun s hd => ?_⟩
  obtain ⟨a, ha⟩ := finite_register s _ hd "a"
    (by simp [LogExp.Atom.lhs, LogExp.Atom.guards, LogExp.productGuards])
  obtain ⟨b, hb⟩ := finite_register s _ hd "b"
    (by simp [LogExp.Atom.lhs, LogExp.Atom.guards, LogExp.productGuards])
  cases backend <;>
    simp [LogExp.Atom.lhs, LogExp.Atom.rhs, LogExp.Backend.log,
      LogExp.splitProduct, LogExp.keepProduct, LogExp.input, LogExp.secondInput,
      LogExp.assignOutput, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, numericLt, numericLe, bop,
      ha, hb, algebra, Software.strictFP32, Software.scalarFP32, Software.fp32Context, rest, State.setReg_same, Function.comp_def]

set_option maxHeartbeats 1600000 in
private theorem log_cancel (log exp : LogExp.Backend) :
    FragmentSound algebra domain (LogExp.Atom.log_exp_cancel log exp).lhs
      (LogExp.Atom.log_exp_cancel log exp).rhs := by
  refine ⟨rfl, fun s hd => ?_⟩
  obtain ⟨a, ha⟩ := finite_register s _ hd "a"
    (by simp [LogExp.Atom.lhs, LogExp.Atom.guards, LogExp.guards])
  cases log <;> cases exp <;>
    simp [LogExp.Atom.lhs, LogExp.Atom.rhs, LogExp.Backend.log, LogExp.Backend.exp,
      LogExp.expression, LogExp.useIdentity, LogExp.absolute, LogExp.input,
      LogExp.assignOutput, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, numericLt, numericLe, bop,
      ha, algebra, Software.strictFP32, Software.scalarFP32, Software.fp32Context, rest, State.setReg_same, Function.comp_def]

private theorem logarithmic_atom (atom : LogExp.Atom) (selected : atom.Available) :
    FragmentSound algebra domain atom.lhs atom.rhs := by
  rcases atom with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _, _ | _⟩ | ⟨_ | _, _ | _⟩
  all_goals first
    | exact log_split _
    | exact log_cancel _ _
    | exact False.elim ((by decide : ¬ _) selected)

private theorem exponential_entries (R : Exponential.Rules) (entry)
    (member : entry ∈ R.assumptions) :
    List.Forall₂ (FragmentSound algebra domain) entry.rule.lhs entry.rule.rhs := by
  rcases List.mem_append.mp member with arithmetic | exponential
  · obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp arithmetic
    unfold ScalarArithmetic.Atom.entry? at he
    split at he
    · cases he
    cases hr : atom.report? with
    | none => simp [hr] at he
    | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (arithmetic_atom atom) .nil
  · obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp exponential
    unfold Exponential.Atom.entry? at he
    split at he
    · cases he
    cases hr : atom.report? with
    | none => simp [hr] at he
    | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (exponential_atom atom) .nil

theorem exponential_models (R : Exponential.Rules) : Models R.assumptions algebra domain :=
  models_of_atoms algebra domain R.assumptions (exponential_entries R) strict_certificates

/-- Every selected arithmetic, exp and log entry holds with its actual guards.
The resulting model covers the complete derivation closure. -/
theorem models (R : LogSumExp.Rules) : Models R.assumptions algebra domain := by
  apply models_of_atoms algebra domain R.assumptions ?_ strict_certificates
  intro entry member
  rcases List.mem_append.mp member with exponential | logarithmic
  · exact exponential_entries R.exponential entry exponential
  · obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp logarithmic
    unfold LogExp.Atom.entry? at he
    cases hr : atom.report? with
    | none => simp [hr] at he
    | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (logarithmic_atom atom (by simp [LogExp.Atom.Available, hr])) .nil

def initial : State Number where
  mem := fun _ _ => .mk .real one
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- The singleton row satisfies every input and intermediate domain check. -/
theorem domain_holds :
    (StableLogSumExpFPContract.requirements "x" 1 Equational.seededSchedules).Holds
      algebra domain initial := by
  rw [StableLogSumExpFPContract.requirements_holds]
  change LogSumExpShift.ShiftDomain algebra domain (fun _ : Fin 1 => one)
    one (.add (.input 0) .zero)
  constructor <;>
    simp [domain, SoftmaxShift.exp, SoftmaxShift.exponentials, SoftmaxShift.shifted,
      SoftmaxShift.scale, ScalarArithmetic.zero, ScalarArithmetic.one, ScalarArithmetic.add,
      ScalarArithmetic.sub, ScalarArithmetic.mul, ScalarArithmetic.div,
      ScalarReduction.value, ScalarReduction.FiniteTree, LogSumExpShift.log, algebra, Software.strictFP32, Software.scalarFP32, Software.fp32Context, rest]
  all_goals decide +kernel

/-- Execute the shared Triton sources, including their bf16 cast and store.
Both runs succeed; the difference is observable at the declared output. -/
theorem stored_outputs :
    Scheduled.observe₁ algebra Equational.seededSchedules initial
      (StableLogSumExpFPContract.stable "x" "y" 1) 0 =
        some (.mk .bf16 one) ∧
    Scheduled.observe₁ algebra Equational.seededSchedules initial
      (StableLogSumExpFPContract.direct "x" "y" 1) 0 =
        some (.mk .bf16 zero) := by
  let engine := SoftmaxStableFPContract.engine algebra Equational.seededSchedules
  obtain ⟨a, ha, hout, _⟩ := StableLogSumExpFPExecution.stable_run engine "x" "y" 1
    (by decide) (fun _ => one) initial (by intro i; rfl)
  obtain ⟨b, hb, hout', _⟩ := StableLogSumExpFPExecution.direct_run engine "x" "y" 1
    (fun _ => one) initial (by intro i; rfl)
  change a.mem "y" 0 = _ at hout
  change b.mem "y" 0 = _ at hout'
  constructor
  · change (Structural.exec engine (StableLogSumExp.Kernels.stableLSEKernel "x" "y" 1)
      initial).map (fun t => t.mem "y" 0) = _
    rw [ha]
    change some (a.mem "y" 0) = _
    rw [hout]
    have hv : StableLogSumExpFPExecution.stableValue
        (SoftmaxStableFPExecution.sourceEngine engine)
        (SoftmaxStableFPExecution.loadedRow engine (fun _ : Fin 1 => one)) = one := by
      simp only [StableLogSumExpFPExecution.stableValue, SoftmaxStableFPExecution.maximum,
        SoftmaxStableFPExecution.sourceEngine, SoftmaxStableFPExecution.loadedRow,
        engine, SoftmaxStableFPContract.engine, Scheduled.fp32, Scheduled.Profile.algebra,
        Algebra.withDefaultPrecision, ScalarReduction.algebra, resolvePrecision,
        Option.getD_none]
      simp [algebra, Software.strictFP32, Software.scalarFP32, Software.fp32Context, rest]
      decide +kernel
    rw [hv]
  · change (Structural.exec engine (StableLogSumExp.Kernels.directLSEKernel "x" "y" 1)
      initial).map (fun t => t.mem "y" 0) = _
    rw [hb]
    change some (b.mem "y" 0) = _
    rw [hout']
    rfl

/-- The stored FP32 carrier words differ even though the source output dtype
and the observed store address agree. -/
theorem stored_outputs_different :
    Scheduled.observe₁ algebra Equational.seededSchedules initial
      (StableLogSumExpFPContract.stable "x" "y" 1) 0 ≠
    Scheduled.observe₁ algebra Equational.seededSchedules initial
      (StableLogSumExpFPContract.direct "x" "y" 1) 0 := by
  rw [stored_outputs.1, stored_outputs.2]
  intro h
  have hv : one = zero :=
    congrArg (fun c : Cell Number => c.read .bf16) (Option.some.inj h)
  exact (by decide +kernel : one ≠ zero) hv

#axiomsClean domain_strict
#axiomsClean strict_certificates
#axiomsClean arithmetic_atom
#axiomsClean exponential_atom
#axiomsClean models
#axiomsClean domain_holds
#axiomsClean stored_outputs
#axiomsClean stored_outputs_different

end VeriTile.Bench.Examples.StableLogSumExpUnconditional.Model
