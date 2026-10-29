import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Contract
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- Accepted finite Python numerics, retaining their actual kind. The Real
field of `floating` denotes a finite binary64 value; nonfinite/rejected
objects are covered by the observation-level checker in Contract.lean. -/
inductive HostScalar where
  | boolean (value : Bool)
  | integer (value : Int)
  | floating (value : ℝ)

def HostScalar.value : HostScalar → ℝ
  | .boolean b => if b then 1 else 0
  | .integer n => n
  | .floating x => x

def HostScalar.isInteger : HostScalar → Bool
  | .boolean _ | .integer _ => Bool.true
  | .floating _ => Bool.false
def HostScalar.isFloat : HostScalar → Bool
  | .floating _ => Bool.true
  | _ => Bool.false

def EpsilonAdmitted : HostScalar → Prop
  | .boolean b => b=Bool.true
  | .integer n => 0<n ∧ n<(2:Int)^64
  | .floating x => 0<x
def ThetaAdmitted : HostScalar → Prop
  | .floating x => Contract.minNormal32 ≤ x ∧ x ≤ Contract.maxFinite32
  | _ => False

/-- The selected Triton 3.7.1 literal inference, before weak scalar promotion
or extern lookup. These are literal types, not approximate value predicates. -/
inductive LiteralType where
  | int1 | int32 | uint32 | int64 | uint64 | float32 | float64
  deriving DecidableEq, Repr

def literalType : HostScalar → Option LiteralType
  | .boolean _ => some .int1
  | .integer n =>
    if -(2:Int)^31 ≤ n ∧ n<(2:Int)^31 then some .int32
    else if (2:Int)^31 ≤ n ∧ n<(2:Int)^32 then some .uint32
    else if -(2:Int)^63 ≤ n ∧ n<(2:Int)^63 then some .int64
    else if (2:Int)^63 ≤ n ∧ n<(2:Int)^64 then some .uint64
    else none
  | .floating x => by
    classical
    exact if x=0 ∨ (Contract.minNormal32 ≤ |x| ∧ |x| ≤ Contract.maxFinite32)
      then some .float32 else some .float64

/-- EPS is a weak scalar operand of a float32 tensor addition. Every supported
numeric literal kind is rematerialized as float32 at that source site. -/
def epsilonType (s : HostScalar) : Option LiteralType :=
  (literalType s).map (fun _ => .float32)

/-- `libdevice.pow` receives a float32 exponent and looks up the exact pair
before promotion. Of this source's possible bases only float32 dispatches. -/
def thetaType (s : HostScalar) : Option LiteralType :=
  match literalType s with
  | some .float32 => some .float32
  | _ => none

theorem epsilon_type (s : HostScalar) (h : EpsilonAdmitted s) :
    epsilonType s=some .float32 := by
  cases s with
  | boolean b => rfl
  | integer n =>
    simp only [EpsilonAdmitted] at h
    simp only [epsilonType,literalType]
    split_ifs <;> simp_all <;> omega
  | floating x =>
    classical
    simp only [epsilonType,literalType]
    split_ifs <;> rfl

theorem theta_type (s : HostScalar) (h : ThetaAdmitted s) :
    thetaType s=some .float32 := by
  cases s with
  | boolean b => exact False.elim h
  | integer n => exact False.elim h
  | floating x =>
    have positive : 0<x := by
      have lo : (0:ℝ)<Contract.minNormal32 := by
        unfold Contract.minNormal32
        positivity
      exact lt_of_lt_of_le lo h.1
    simp [thetaType,literalType,abs_of_pos positive,h.1,h.2]

/-- Local scalar-literal realization ports retain the Python kind. Integer
and finite-binary64 materialization may round differently; neither is routed
through a single Real-to-word compatibility function. EPS may materialize as
zero or infinity. These are scalar numerical boundaries only. -/
structure ScalarMaterialization (α : Type) where
  boolean : Bool → α
  integer : Int → α
  floating : ℝ → α

def ScalarMaterialization.apply (a : ScalarMaterialization α) : HostScalar → α
  | .boolean b => a.boolean b
  | .integer n => a.integer n
  | .floating x => a.floating x

structure Scalars (α : Type) where
  epsilon : α
  theta : α

def materializedScalars (a : ScalarMaterialization α) (epsilon theta : HostScalar) : Scalars α :=
  ⟨a.apply epsilon,a.apply theta⟩

def bindScalars (scalars : Scalars α) (s : State α) : State α :=
  (s.setReg "EPS" .real [] (fun _ => scalars.epsilon)).setReg
    "THETA" .real [] (fun _ => scalars.theta)

theorem scalar_registers (a : ScalarMaterialization α) (epsilon theta : HostScalar)
    (s : State α) :
    (bindScalars (materializedScalars a epsilon theta) s).regs .real [] "EPS"=
      some (fun _ => a.apply epsilon) ∧
    (bindScalars (materializedScalars a epsilon theta) s).regs .real [] "THETA"=
      some (fun _ => a.apply theta) := by
  simp [bindScalars,materializedScalars,State.setReg]

/-- Relate literal observations to the same exact wrapper metadata. This
mentions only argument kinds and values, never source execution or output. -/
def ScalarObservations (i : Contract.Inputs) (epsilon theta : HostScalar) : Prop :=
  epsilon.value=i.original.scalars.epsilon ∧ theta.value=i.original.scalars.theta ∧
  epsilon.isInteger=i.epsilonInteger ∧ theta.isFloat=i.thetaFloat

theorem epsilon_admitted_iff (s : HostScalar) :
    EpsilonAdmitted s ↔ 0<s.value ∧
      (s.isInteger=Bool.true → s.value<(2:ℝ)^64) := by
  cases s with
  | boolean b => cases b <;> norm_num [EpsilonAdmitted,HostScalar.value,HostScalar.isInteger]
  | integer n =>
    simp only [EpsilonAdmitted,HostScalar.value,HostScalar.isInteger,true_implies]
    norm_cast
  | floating x => simp [EpsilonAdmitted,HostScalar.value,HostScalar.isInteger]

theorem theta_admitted_iff (s : HostScalar) :
    ThetaAdmitted s ↔ s.isFloat=Bool.true ∧
      Contract.minNormal32 ≤ s.value ∧ s.value ≤ Contract.maxFinite32 := by
  cases s <;> simp [ThetaAdmitted,HostScalar.value,HostScalar.isFloat]

/-- The candidate's successful host checks imply the actual scalar dispatch
requirements; these requirements are not extra narrowed theorem premises. -/
theorem checked_scalar_domain (i : Contract.Inputs) (h : Contract.InputContract i)
    (epsilon theta : HostScalar) (observations : ScalarObservations i epsilon theta) :
    EpsilonAdmitted epsilon ∧ ThetaAdmitted theta := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,_,scalars,_⟩
  rcases scalars with ⟨_,_,_,_,_,_,_,_,_,_,_,positive,_,_,_⟩
  rcases observations with ⟨he,ht,hi,hf⟩
  constructor
  · apply (epsilon_admitted_iff epsilon).mpr
    rw [he,hi]
    exact ⟨positive,h.2.1⟩
  · apply (theta_admitted_iff theta).mpr
    rw [ht,hf]
    exact h.2.2

theorem checked_scalar_types (i : Contract.Inputs) (h : Contract.InputContract i)
    (epsilon theta : HostScalar) (observations : ScalarObservations i epsilon theta) :
    epsilonType epsilon=some .float32 ∧ thetaType theta=some .float32 := by
  obtain ⟨he,ht⟩ := checked_scalar_domain i h epsilon theta observations
  exact ⟨epsilon_type epsilon he,theta_type theta ht⟩

#axiomsClean epsilon_type
#axiomsClean theta_type
#axiomsClean scalar_registers
#axiomsClean checked_scalar_domain
#axiomsClean checked_scalar_types
#specNonCircular ScalarObservations avoiding [bindScalars,materializedScalars]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
