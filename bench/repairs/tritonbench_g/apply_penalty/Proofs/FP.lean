import bench.repairs.tritonbench_g.apply_penalty.Proofs.Execution
import VeriTile.Triton.Float.BitValue
import VeriTile.Triton.Float.ScalarOps
import Mathlib.Tactic.Ring

/-! Optional historical FP32 transport, adapted from the accepted whole003 local schedule.
The original native frequency operation is FMA, never identified with separately
rounded multiply/subtract. These explicit-premise transport results are not
current numerical experiment obligations. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.FPProof
open VeriTile.Triton
noncomputable section

abbrev Word := FP.Value .fp32

structure NativeOps where
  zero : Word
  fromInt32 : BitVec 32 → Word
  positive : Word → Bool
  divide : Word → Word → Word
  multiply : Word → Word → Word
  subtract : Word → Word → Word
  flipSign : Word → Word
  fma : Word → Word → Word → Word

def NativeOps.scalar (ops : NativeOps) : ScalarOps Word where
  zero := ops.zero
  fromInt32 := ops.fromInt32
  positive := ops.positive
  divide := ops.divide
  multiply := fun x r => ops.multiply r x
  subtract := ops.subtract
  subtractProduct := fun selected count frequency => ops.fma (ops.flipSign frequency) count selected

/-- Independent ordered FP32 target. Comparison, int32 conversion, division,
product, frequency sign flip, fused operation and final subtraction stay opaque. -/
def nativePenalty (ops : NativeOps) (x p f r : Word) (c : BitVec 32) : Word :=
  let q := ops.divide x r
  let m := ops.multiply r x
  let selected := if ops.positive x then q else m
  ops.subtract (ops.fma (ops.flipSign f) (ops.fromInt32 c) selected) p

theorem arithmetic_native (ops : NativeOps) (x p f r : Word) (c : BitVec 32) :
    (arithmetic ops.scalar x p f r c).result=nativePenalty ops x p f r c := rfl

/-- `div.full.f32` is a target-specific division profile. It is deliberately
not replaced by IEEE correctly rounded division. The other inspected native
instructions use RNE binary32 with input/result FTZ disabled. -/
structure RoundedProfile where
  divFull32 : Word → Word → Word

def roundedOps (profile : RoundedProfile) : NativeOps where
  zero := FP.Value.zero .fp32
  fromInt32 := fun count => FP.round .fp32 .ieeeValue (Rat.ofInt count.toInt)
  positive := fun x => FP.lt .ieeeValue (FP.Value.zero .fp32) x
  divide := profile.divFull32
  multiply := FP.mul .ieeeValue
  subtract := FP.sub .ieeeValue
  flipSign := FP.Value.neg
  fma := FP.fma .ieeeValue

/-- Meaningful native-conformance ports at the actual evaluated operands.
Finite entry operands and a nonzero repetition scalar avoid assuming a NaN
payload policy. Overflow/subnormal rounding remains part of the rounded word
reference, and int32 values above 2^24 are rounded rather than assumed exact. -/
structure LocalRounded (ops : NativeOps) (profile : RoundedProfile)
    (x p f r : Word) (count : BitVec 32) : Prop where
  inputsFinite : x.isNaN=Bool.false ∧ x.isInf=Bool.false ∧
    p.isNaN=Bool.false ∧ p.isInf=Bool.false ∧ f.isNaN=Bool.false ∧ f.isInf=Bool.false ∧
    r.isNaN=Bool.false ∧ r.isInf=Bool.false
  divisorNonzero : r.isZero=Bool.false
  countRNE : ops.fromInt32 count=(roundedOps profile).fromInt32 count
  compareOrdered : ops.positive x=(roundedOps profile).positive x
  divideFull : ops.divide x r=profile.divFull32 x r
  multiplyRNE : ops.multiply r x=FP.mul .ieeeValue r x
  signFlip : ops.flipSign f=f.neg
  fusedRNE : let selected := if ops.positive x then ops.divide x r else ops.multiply r x
    ops.fma (ops.flipSign f) (ops.fromInt32 count) selected=
      FP.fma .ieeeValue (ops.flipSign f) (ops.fromInt32 count) selected
  subtractRNE : let selected := if ops.positive x then ops.divide x r else ops.multiply r x
    let weighted := ops.fma (ops.flipSign f) (ops.fromInt32 count) selected
    ops.subtract weighted p=FP.sub .ieeeValue weighted p

theorem native_local_rounded (ops : NativeOps) (profile : RoundedProfile)
    (x p f r : Word) (count : BitVec 32) (h : LocalRounded ops profile x p f r count) :
    nativePenalty ops x p f r count=nativePenalty (roundedOps profile) x p f r count := by
  unfold nativePenalty
  rw [h.subtractRNE,h.fusedRNE,h.signFlip,h.countRNE,h.compareOrdered,h.divideFull,h.multiplyRNE]
  rfl

/-- Optional exact-arithmetic lemma hypotheses. These are NOT deferred native
conformance obligations: universal exact-real count conversion/arithmetic is
known false. Both evaluated arms and actual conversion are recorded. -/
structure LocalExact (ops : NativeOps) (decode : Word → ℝ) (finite : Word → Prop)
    (x p f r : Word) (count : BitVec 32) : Prop where
  inputsFinite : finite x ∧ finite p ∧ finite f ∧ finite r
  divisorNonzero : decode r ≠ 0
  countExact : decode (ops.fromInt32 count) = (count.toInt : ℝ)
  signExact : ops.positive x = decide (0 < decode x)
  divideExact : decode (ops.divide x r) = decode x/decode r
  multiplyExact : decode (ops.multiply r x) = decode x*decode r
  bothArmsFinite : finite (ops.divide x r) ∧ finite (ops.multiply r x)
  signFlipExact : decode (ops.flipSign f) = -decode f
  fusedExact : let selected := if ops.positive x then ops.divide x r else ops.multiply r x
    decode (ops.fma (ops.flipSign f) (ops.fromInt32 count) selected) =
      decode (ops.flipSign f)*decode (ops.fromInt32 count)+decode selected
  fusedFinite : let selected := if ops.positive x then ops.divide x r else ops.multiply r x
    finite (ops.fma (ops.flipSign f) (ops.fromInt32 count) selected)
  finalSubtractExact : let selected := if ops.positive x then ops.divide x r else ops.multiply r x
    let weighted := ops.fma (ops.flipSign f) (ops.fromInt32 count) selected
    decode (ops.subtract weighted p)=decode weighted-decode p

theorem native_local_real (ops : NativeOps) (decode : Word → ℝ) (finite : Word → Prop)
    (x p f r : Word) (count : BitVec 32) (h : LocalExact ops decode finite x p f r count) :
    decode (nativePenalty ops x p f r count)=
      penalty (decode x) count.toInt (decode p) (decode f) (decode r) := by
  unfold nativePenalty
  rw [h.finalSubtractExact,h.fusedExact,h.countExact,h.signFlipExact]
  by_cases hp : 0 < decode x
  · simp [penalty,h.signExact,hp,h.divideExact]; ring
  · simp [penalty,h.signExact,hp,h.multiplyExact]; ring

/-- Optional exact-arithmetic domain, not a native experimental target. -/
def NumericalDomain (ops : NativeOps) (a : Arguments) (input : Inputs Word)
    (decode : Word → ℝ) (finite : Word → Prop) : Prop :=
  ∀ b,b<a.batches → ∀ lane,lane<a.block →
    let active := a.start b+lane<a.stop b
    LocalExact ops decode finite
      (if active then input.logits (address a b lane) else ops.zero)
      (input.presence b) (input.frequency b) (input.repetition b)
      (if active then a.counts (a.start b+lane) else 0)

#axiomsClean arithmetic_native
#axiomsClean native_local_rounded
#axiomsClean native_local_real
end
end VeriTile.Bench.Repairs.ApplyPenalty.FPProof
