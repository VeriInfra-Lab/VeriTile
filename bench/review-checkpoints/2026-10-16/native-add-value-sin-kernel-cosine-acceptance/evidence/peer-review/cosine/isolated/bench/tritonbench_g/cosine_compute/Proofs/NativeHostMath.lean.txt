import bench.tritonbench_g.cosine_compute.Proofs.NativeScalar

namespace CosineNative.HostMath
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Nearest integer quotient with ties to the even quotient. -/
def nearestEven (a denominator : Nat) : Nat :=
  let q := a / denominator
  let r := a % denominator
  q + if 2*r>denominator ∨ (2*r=denominator ∧ q%2=1) then 1 else 0

/-- Finite specification of nearest-even integer-to-binary64 conversion on
nonnegative signed64 input. Native CPython correspondence is explicitly
deferred in the authority records; sampling is not a refinement proof. The
specification cannot produce a fraction or infinity in this host domain. -/
def integer64 (n : Nat) : Nat :=
  let shift := n.log2-52
  nearestEven n (2^shift) * 2^shift

/-- The correctly-rounded nearest-even binary64 square-root specification
uses 53 significand bits. Native math.sqrt/libm correspondence, including the
current floating environment, is explicitly deferred in authority records. For signed64 numel the
unbiased result exponent is at most31, so its significand is the nearest-even
integer square root of this exactly scaled integer. The midpoint comparison
is performed by squaring, without any real-valued or native callback oracle. -/
def sqrtExponent (n : Nat) : Nat := (integer64 n).log2/2
def sqrtScale (n : Nat) : Nat := 52-sqrtExponent n
def sqrtSignificand (n : Nat) : Nat :=
  let square := integer64 n * 2^(2*sqrtScale n)
  let lower := square.sqrt
  let midpoint := (2*lower+1)^2
  lower + if 4*square>midpoint ∨ (4*square=midpoint ∧ lower%2=1) then 1 else 0

/-- Exact ceiling of the represented binary64 square root. -/
def ceilSqrt (n : Nat) : Nat :=
  if n=0 then 0 else
  (sqrtSignificand n+2^(sqrtScale n)-1)/2^(sqrtScale n)

/-- Installed Triton next_power_of_2's exact six shifts. Its Python input-zero
branch produces zero, since Python's intermediate -1 remains -1 after shifts. -/
def nextPower (n : Nat) : Nat :=
  if n=0 then 0 else
  let x := n-1
  let x := x ||| (x >>> 1)
  let x := x ||| (x >>> 2)
  let x := x ||| (x >>> 4)
  let x := x ||| (x >>> 8)
  let x := x ||| (x >>> 16)
  let x := x ||| (x >>> 32)
  x+1

def block (n : Nat) : Nat := nextPower (ceilSqrt n)
def grid (n : Nat) : Nat := (n+block n-1)/block n

inductive Failure where | gridDivisionByZero
  deriving DecidableEq, Repr
structure Shape where
  n : Nat
  block : Nat
  grid : Nat
  deriving DecidableEq, Repr

/-- cdiv(0,0) is a host exception after empty_like and the math steps. Lean's
total Nat division is never used to fabricate a zero-grid host success. -/
def prepare (n : Nat) : Except Failure Shape :=
  if block n=0 then .error .gridDivisionByZero else .ok ⟨n,block n,grid n⟩

theorem zero_fails : prepare 0=.error .gridDivisionByZero := by decide +kernel
theorem ordinary_profile : prepare 4096=.ok ⟨4096,64,64⟩ := by decide +kernel
theorem two31_profile : prepare (2^31)=.ok ⟨2^31,65536,32768⟩ := by decide +kernel
theorem largest_admitted_boundary : block (2^40)=2^20 ∧ block (2^40+1)=2^21 := by decide +kernel
theorem rounded_boundary : block (2^52+1)=2^26 ∧ block (2^52+2)=2^27 := by decide +kernel
theorem maximum_profile : block (2^63-1)=2^32 := by decide +kernel

#axiomsClean zero_fails
#axiomsClean ordinary_profile
#axiomsClean two31_profile
#axiomsClean largest_admitted_boundary
#axiomsClean rounded_boundary
#axiomsClean maximum_profile
#guard_msgs (drop info) in
#auditModuleAxioms
end CosineNative.HostMath
