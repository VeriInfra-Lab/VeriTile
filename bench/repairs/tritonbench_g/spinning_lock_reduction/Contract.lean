import bench.repairs.tritonbench_g.spinning_lock_reduction.Kernels

/-! Pure observations and executable admission. Tensor descriptors and raw words
are observations of the three caller-owned tensors; no buffers are allocated.
Negative Python integers are rejected before conversion to natural indices. -/
namespace VeriTile.Bench.Repairs.SpinningLock.Admission
open VeriTile.Bench.TritonBenchG.SpinningLock

structure Scalars where
  pythonInts : Bool
  sms : Int
  group : Int
  m : Int
  n : Int
  sm : Int
  sn : Int
  bm : Int
  bn : Int

def Scalars.args (s : Scalars) : HostArgs :=
  ⟨s.sms.toNat,s.group.toNat,s.m.toNat,s.n.toNat,s.sm.toNat,s.sn.toNat,s.bm.toNat,s.bn.toNat⟩

def scalarCheck (s : Scalars) : Bool := s.pythonInts && decide
  (0 ≤ s.sms ∧ 0 ≤ s.group ∧ 0 ≤ s.m ∧ 0 ≤ s.n ∧
   0 ≤ s.sm ∧ 0 ≤ s.sn ∧ 0 ≤ s.bm ∧ 0 ≤ s.bn)

inductive DType where | fp32 | int32 | other deriving DecidableEq
structure Tensor where
  object : Nat
  device : Nat
  shape : List Nat
  strides : List Nat
  base : Nat
  isTensor : Bool
  strided : Bool
  cuda : Bool
  contiguous : Bool
  ordinary : Bool
  dtype : DType
  word : Nat → BitVec 32

def Tensor.elements (t : Tensor) : Nat := t.shape.foldl (· * ·) 1

def tensorCheck (t : Tensor) (dtype : DType) (device : Nat) : Bool :=
  t.isTensor && t.strided && t.cuda && t.contiguous && t.ordinary &&
    decide (t.dtype = dtype ∧ t.device = device)

def Window (t : Tensor) : Prop :=
  4*t.elements < 2^63 ∧ t.base%4=0 ∧ t.base+4*t.elements≤2^64
instance (t : Tensor) : Decidable (Window t) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))
def Disjoint (a b : Tensor) : Prop :=
  a.base+4*a.elements≤b.base ∨ b.base+4*b.elements≤a.base
instance (a b : Tensor) : Decidable (Disjoint a b) := inferInstanceAs (Decidable (_ ∨ _))

def Metadata (h : HostArgs) (p c l : Tensor) : Prop :=
  0<h.numSms ∧ h.numSms<2^31 ∧ 0<h.groupK ∧ h.groupK<2^64 ∧
  0<h.bm ∧ h.bm≤1048576 ∧ 0<h.bn ∧ h.bn≤1048576 ∧
  h.bm &&& (h.bm-1)=0 ∧ h.bn &&& (h.bn-1)=0 ∧ h.bm*h.bn≤1048576 ∧
  h.m<2^63 ∧ h.n<2^63 ∧ h.strideM<2^63 ∧ h.strideN<2^63 ∧
  c.shape=[h.m,h.n] ∧ c.strides=[h.strideM,h.strideN] ∧ l.shape.length=1 ∧
  h.numSms*h.bm*h.bn≤p.elements ∧ h.numSms≤l.elements ∧
  (h.m=0 ∨ h.n=0 ∨ (h.m-1)*h.strideM+(h.n-1)*h.strideN<c.elements) ∧
  Window p ∧ Window c ∧ Window l ∧ Disjoint p c ∧ Disjoint p l ∧ Disjoint c l ∧
  (h.bm-1)*h.strideM+(h.numSms*h.bn-1)*h.strideN<2^63 ∧
  c.base+4*((h.bm-1)*h.strideM+(h.numSms*h.bn-1)*h.strideN)+4≤2^64
instance (h : HostArgs) (p c l : Tensor) : Decidable (Metadata h p c l) :=
  inferInstanceAs (Decidable (_ ∧ _))

def readyCheck (h : HostArgs) (p l : Tensor) : Bool :=
  (List.range l.elements).all fun q =>
    if l.word q == 1 then
      decide ((q+1)*(h.bm*h.bn)≤p.elements) &&
      (List.range (h.bm*h.bn)).all (fun lane => p.word (q*(h.bm*h.bn)+lane) == 0)
    else Bool.true

def check (s : Scalars) (p c l : Tensor) : Bool :=
  scalarCheck s && tensorCheck p .fp32 p.device && tensorCheck c .fp32 p.device &&
  tensorCheck l .int32 p.device && decide (Metadata s.args p c l) && readyCheck s.args p l

/-- Finite tensor observations extended outside their windows. No kernel access
uses the extension; bounds are proved separately from the executable guard. -/
def memory (decode : BitVec 32 → V) (p c l : Tensor) : Memory V :=
  ⟨fun a => decode (p.word a.toNat), fun a => decode (c.word a.toNat),
   fun a => if 0≤a ∧ a.toNat<l.elements then l.word a.toNat else 0⟩
end VeriTile.Bench.Repairs.SpinningLock.Admission
