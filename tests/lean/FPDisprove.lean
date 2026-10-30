import VeriTile.Triton.Float.Disprove
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

/-! Two independent source kernels test model checks, input contracts, lane
search, casts and comparison availability. The model satisfies the fixed
strict FP32 software theory; mixed-format casts remain in its arbitrary
residual backend. No StableLogSumExp imports or facts. -/
noncomputable section
namespace VeriTile.Tests.FPDisprove
open VeriTile Triton FP FP.Structural FP.Guarded
open scoped VeriTile.Spec

def writeZero (y : RegionName) : ComputeKernel := triton {
  zero := 0.0
  tl.store($(y) + 0, (zero).to(tl.bfloat16))
  tl.store($(y) + 1, (zero).to(tl.bfloat16))
}

def writeOne (y : RegionName) : ComputeKernel := triton {
  zero := 0.0
  one := 1.0
  tl.store($(y) + 0, (zero).to(tl.bfloat16))
  tl.store($(y) + 1, (one).to(tl.bfloat16))
}

def original : Scheduled.IO₁ where
  io := {
    kernel := writeZero "y", projection := by rfl,
    inp := "x", out := "y", Bin := 1, Bout := 2,
    read := id, write := fun _ => 0 }
  profile := Scheduled.fp32
  domain := fun _ => .all []

def optimized : Scheduled.IO₁ :=
  { original with io := original.io.withKernel (writeOne "y") }

def zero : FP.Value .fp32 := FP.Value.zero .fp32
def one : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000

/-- Only primitives outside the strict FP32 profile can vary between models.
In particular the tested BF16 store casts deliberately differ. -/
def residual (eraseCast : Bool) : Algebra (FP.Value .fp32) where
  literal := fun _ _ r => if r = 0 then zero else one
  negInf := zero
  binary := fun _ _ _ a _ => a
  unary := fun _ _ => id
  cast := fun _ _ _ a => if eraseCast then zero else a
  fromNat := fun _ _ => zero
  fromInt := fun _ _ => zero
  fp32Bits := fun b => ⟨b.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => zero
  reduceSum := fun _ {_} _ _ _ _ => zero

def model (eraseCast : Bool := Bool.false) : Algebra (FP.Value .fp32) :=
  Software.strictFP32 (residual eraseCast)

def domain : Domain (FP.Value .fp32) := strictDomain

theorem models (eraseCast : Bool) : Models [] (model eraseCast) domain :=
  strict_models_empty (residual eraseCast)

/-- The source programs ignore the input. Its actual NaN value lets the
finite-input negative test fail on its contract while output observations
still succeed; no fabricated empty-domain model is needed. -/
def initial : State (FP.Value .fp32) where
  mem := fun _ _ => .mk .real (FP.Value.qNaN .fp32)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem domain_holds :
    (original.domain Equational.seededSchedules).Holds model domain initial := by
  simp [original, GuardExpression.Condition.Holds]

set_option linter.unusedSimpArgs false in
theorem observations (eraseCast : Bool) (i : Fin 2) :
    Scheduled.observe₁ (model eraseCast) Equational.seededSchedules initial original i.val =
      some (.mk .bf16 zero) ∧
    Scheduled.observe₁ (model eraseCast) Equational.seededSchedules initial optimized i.val =
      some (.mk .bf16 (if i.val = 1 && !eraseCast then one else zero)) := by
  rcases i with ⟨i, hi⟩
  have cases : i = 0 ∨ i = 1 := by omega
  rcases cases with rfl | rfl <;> cases eraseCast <;>
    norm_num [Scheduled.observe₁, original, optimized, writeZero, writeOne,
      Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold, store,
      State.setReg, State.write, Scheduled.Profile.algebra, Scheduled.fp32,
      ScalarReduction.algebra, Algebra.withDefaultPrecision, model, initial,
      Software.strictFP32, Software.scalarFP32, Software.fp32Context, residual, zero, one,
      TileShape.allIndices, Region.cast, bop, Function.comp_def, ofFloat, toFloat]
  all_goals simp [Option.bind, FloatDType.toTileDType]

/-- The opaque cell carrier has no decision procedure for equality; expose
the checked inequality of the two observed software bit values to search. -/
theorem stored_values_differ :
    (some (Cell.mk .bf16 zero) : Option (Cell (FP.Value .fp32))) ≠
      some (Cell.mk .bf16 one) := by
  intro h
  have values := congrArg (fun cells : Option (Cell (FP.Value .fp32)) =>
    cells.map (fun cell => cell.read .bf16)) h
  have different : zero ≠ one := by decide +kernel
  exact different (by simpa only [Option.map_some, Cell.read_same, Option.some.injEq] using values)

-- Lane zero agrees. The tactic must backtrack and find the second lane.
theorem second_lane_differs : ¬ (original ≡[([] : Spec.Assumptions GuardedFragment)] optimized) := by
  have different := stored_values_differ
  disprove [models Bool.false, domain_holds, observations Bool.false, initial, Equational.seededSchedules]

-- The first model erases the difference; search must try the next model.
theorem second_model_differs : ¬ (original ≡[([] : Spec.Assumptions GuardedFragment)] optimized) := by
  have different := stored_values_differ
  disprove [models Bool.true, models Bool.false, domain_holds,
    observations Bool.true, observations Bool.false, initial, Equational.seededSchedules]

def conditionalOriginal : FP.Kernel₁ := { original with comparisons := [] }
def conditionalOptimized : FP.Kernel₁ := { optimized with comparisons := [] }

theorem public_interface :
    ¬ (conditionalOriginal ≡[([] : Spec.Assumptions GuardedFragment)] conditionalOptimized) := by
  have different := stored_values_differ
  disprove [models Bool.false, domain_holds, observations Bool.false, initial, Equational.seededSchedules]

-- A witness in a different theory is not a model of the requested assumptions.
set_option linter.unusedVariables false in
theorem missing_model (R : Spec.Assumptions GuardedFragment) : True := by
  have different := stored_values_differ
  fail_if_success
    have : ¬ (original ≡[R] optimized) := by
      disprove [models Bool.false, domain_holds, observations Bool.false, initial, Equational.seededSchedules]
  trivial

-- No output difference remains after this model's store cast.
theorem cast_difference_is_required : True := by
  have different := stored_values_differ
  have hd : (original.domain Equational.seededSchedules).Holds (model Bool.true) domain initial :=
    domain_holds
  fail_if_success
    have : ¬ (original ≡[([] : Spec.Assumptions GuardedFragment)] optimized) := by
      disprove [models Bool.true, hd, observations Bool.true, initial, Equational.seededSchedules]
  trivial

def requireComparison (p : Scheduled.IO₁) : FP.Kernel₁ :=
  { p with comparisons := [.lt .fp32 .real] }

theorem comparisons_are_required : True := by
  have different := stored_values_differ
  fail_if_success
    have : ¬ (requireComparison original ≡[([] : Spec.Assumptions GuardedFragment)]
        requireComparison optimized) := by
      disprove [models Bool.false, domain_holds, observations Bool.false, initial, Equational.seededSchedules]
  trivial

theorem equal_program_is_not_refuted : True := by
  have different := stored_values_differ
  fail_if_success
    have : ¬ (original ≡[([] : Spec.Assumptions GuardedFragment)] original) := by
      disprove [models Bool.false, domain_holds, observations Bool.false, initial, Equational.seededSchedules]
  trivial

-- A stored difference outside the declared output window is not a witness.
def firstLaneOnly (p : Scheduled.IO₁) : Scheduled.IO₁ :=
  { p with io := { p.io with Bout := 1 } }

theorem output_bound_is_required : True := by
  have different := stored_values_differ
  fail_if_success
    have : ¬ (firstLaneOnly original ≡[([] : Spec.Assumptions GuardedFragment)]
        firstLaneOnly optimized) := by
      disprove [models Bool.false, domain_holds, observations Bool.false,
        initial, Equational.seededSchedules]
  trivial

def finiteInput (p : Scheduled.IO₁) : Scheduled.IO₁ :=
  { p with domain := fun _ => .guard .finite (.input ⟨"x", id, .real⟩) }

theorem finite_input_contract_is_false :
    ¬ ((finiteInput original).domain Equational.seededSchedules).Holds model domain initial := by
  change ¬ ((FP.Value.qNaN .fp32).toRat?.isSome = Bool.true)
  decide +kernel

theorem domain_is_required : True := by
  have different := stored_values_differ
  fail_if_success
    have : ¬ (finiteInput original ≡[([] : Spec.Assumptions GuardedFragment)]
        finiteInput optimized) := by
      disprove [models Bool.false, observations Bool.false, initial, Equational.seededSchedules]
  trivial

theorem signature_difference :
    ¬ (original ≡[([] : Spec.Assumptions GuardedFragment)] firstLaneOnly original) := by
  disprove

#axiomsClean second_lane_differs
#axiomsClean public_interface
#axiomsClean second_model_differs
#axiomsClean signature_difference
#axiomsClean models
#axiomsClean stored_values_differ
#axiomsClean finite_input_contract_is_false

end VeriTile.Tests.FPDisprove
