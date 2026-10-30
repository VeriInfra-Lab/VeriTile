import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.InputContract

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Contract
open RmsMatmulRbe.Contract
noncomputable section
set_option maxHeartbeats 3000000

/-- Observations of the two additional Python predicates. `epsilonInteger`
is `isinstance(eps, int)` (including bool); `thetaFloat` is
`isinstance(theta, float)`. Existing numeric/finite/value observations and
all tensor metadata remain the original contract's observations. -/
structure Inputs where
  original : RmsMatmulRbe.Contract.Inputs
  epsilonInteger : Bool
  thetaFloat : Bool

def minNormal32 : ℝ := 1 / (2:ℝ)^126
def maxFinite32 : ℝ := (2 - 1 / (2:ℝ)^23) * (2:ℝ)^127

/-- EPS weak scalar materialization and THETA extern dispatch are different
source sites. In particular there is no finite-FP32-result condition on EPS. -/
def DispatchAdmitted (i : Inputs) : Prop :=
  (i.epsilonInteger=Bool.true → i.original.scalars.epsilon<(2:ℝ)^64) ∧
  i.thetaFloat=Bool.true ∧
  minNormal32 ≤ i.original.scalars.theta ∧ i.original.scalars.theta ≤ maxFinite32

def dispatchGuard (i : Inputs) : Bool := by
  classical
  exact !(i.epsilonInteger && decide (i.original.scalars.epsilon≥(2:ℝ)^64)) &&
    i.thetaFloat && decide (minNormal32 ≤ i.original.scalars.theta) &&
    decide (i.original.scalars.theta ≤ maxFinite32)

def InputContract (i : Inputs) : Prop :=
  RmsMatmulRbe.Contract.InputContract i.original ∧ DispatchAdmitted i

/-- Exact successful-check order of the candidate wrapper. The two new scalar
checks occur inside `_checked_scalars`, before the remaining shape/alias checks. -/
def checkInputs (input : Inputs) : Except Error Unit :=
  let i := input.original
  if !tensorGuard i.x [fp32] || !(i.others.all fun t => tensorGuard t weightTypes) then .error .tensor
  else if !(i.others.all fun t => decide (t.metadata.device=i.x.metadata.device)) then .error .device
  else if !(i.sameFormat.all fun t => decide (t.metadata.dtype=i.qw.metadata.dtype)) then .error .dtype
  else if i.x.metadata.shape.length≠3 ∨ i.qw.metadata.shape.length≠2 ∨
    i.kw.metadata.shape≠i.qw.metadata.shape ∨ i.vw.metadata.shape≠i.qw.metadata.shape then .error .shape
  else if !scalarGuard i.scalars (dim i.x 0) (dim i.x 1) Bool.true || !dispatchGuard input then .error .scalar
  else if dim i.x 2=0 ∨ dim i.qw 1≠dim i.x 2 ∨ i.rms.metadata.shape≠[dim i.x 2] ∨
    dim i.qw 0≠i.scalars.heads.toNat*i.scalars.dim.toNat ∨
    i.k.metadata.numel≠dim i.x 0*dim i.x 1*dim i.qw 0 ∨
    i.v.metadata.numel≠dim i.x 0*dim i.x 1*dim i.qw 0 then .error .shape
  else if !pythonReadFormatGuard i.reads || decide (overlap i.k i.v) ||
    (i.reads.any fun t => decide (overlap i.k t) || decide (overlap i.v t)) then .error .alias
  else .ok ()

theorem dispatchGuard_iff (i : Inputs) : dispatchGuard i=Bool.true ↔ DispatchAdmitted i := by
  classical
  cases h : i.epsilonInteger <;> simp [dispatchGuard,DispatchAdmitted,h,and_assoc]

theorem checkInputs_iff (i : Inputs) : checkInputs i=.ok () ↔ InputContract i := by
  classical
  cases h : dispatchGuard i with
  | false =>
    have hd : ¬DispatchAdmitted i := by
      intro d
      have ht := (dispatchGuard_iff i).mpr d
      simp [h] at ht
    simp [checkInputs,h,InputContract,hd,ite_eq_iff]
  | true =>
    have hd := (dispatchGuard_iff i).mp h
    simpa [checkInputs,RmsMatmulRbe.Contract.checkInputs,h,InputContract,hd]
      using RmsMatmulRbe.Contract.checkInputs_iff i.original

theorem original_contract (i : Inputs) (h : InputContract i) :
    RmsMatmulRbe.Contract.InputContract i.original := h.1

#axiomsClean dispatchGuard_iff
#axiomsClean checkInputs_iff
#axiomsClean original_contract
#specNonCircular InputContract avoiding [checkInputs]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Contract
