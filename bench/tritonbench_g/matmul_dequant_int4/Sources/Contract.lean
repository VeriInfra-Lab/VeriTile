import bench.tritonbench_g.matmul_dequant_int4.Sources.HelperHost

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Physical HelperHost Helpers
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Primitive observations supplement the literal dimensions, strides, dtype,
device, offset, flags, storage base and capacity in Host.Tensor. -/
structure Info where
  tensor : Host.Tensor
  exactTensor : Bool
  rank : Nat
  cuda : Bool
  strided : Bool
  contiguous : Bool
  knownDtype : Bool

def Ordinary (i : Info) (dtype : Host.DType) (rows columns device : Nat) : Prop :=
  i.exactTensor=true ∧ i.rank=2 ∧ i.cuda=true ∧ i.strided=true ∧ i.contiguous=true ∧ i.knownDtype=true ∧
  i.tensor.Ordinary dtype rows columns device ∧ i.tensor.base%dtype.width=0 ∧ i.tensor.base+2^33<2^64

inductive Error where
  | tensor | geometry | group | device | overlap | indices | pointer | finite | scale
  deriving DecidableEq, Repr

def checkOrdinary (i : Info) (dtype : Host.DType) (rows columns device : Nat) : Except Error Unit :=
  if Ordinary i dtype rows columns device then .ok () else .error .tensor

theorem checkOrdinary_iff (i : Info) (dtype : Host.DType) (rows columns device : Nat) :
    checkOrdinary i dtype rows columns device=.ok () ↔ Ordinary i dtype rows columns device := by
  simp [checkOrdinary]

structure PackedInfo where
  weight : Info
  scale : Info
  zero : Info
  group : Nat
  groupIsInt : Bool

def packed (p : PackedInfo) (m : Nat) : PackedArguments :=
  ⟨⟨m,p.weight.tensor.columns,p.weight.tensor.rows*8,p.group⟩,p.weight.tensor,p.scale.tensor,p.zero.tensor⟩

/-- Host-independent value of the actual parsed primitive observations. -/
def PackedFlags (p : PackedInfo) (m : Nat) (dtype : Host.DType) : Prop :=
  Ordinary p.weight .int32 ((packed p m).shape.k/8) (packed p m).shape.n p.weight.tensor.device ∧
  Ordinary p.scale dtype ((packed p m).shape.k/p.group) (packed p m).shape.n p.weight.tensor.device ∧
  Ordinary p.zero .int32 ((packed p m).shape.k/p.group) ((packed p m).shape.n/8) p.weight.tensor.device

def PackedAccepted (p : PackedInfo) (m : Nat) (dtype : Host.DType) : Prop :=
  p.groupIsInt=true ∧ PackedContract (packed p m) dtype ∧ PackedFlags p m dtype

def checkPacked (p : PackedInfo) (m : Nat) (dtype : Host.DType) : Except Error Unit :=
  if p.groupIsInt≠true then .error .group
  else if ¬PackedContract (packed p m) dtype then .error .geometry
  else if ¬PackedFlags p m dtype then .error .tensor
  else .ok ()

theorem checkPacked_iff (p : PackedInfo) (m : Nat) (dtype : Host.DType) :
    checkPacked p m dtype=.ok () ↔ PackedAccepted p m dtype := by
  unfold checkPacked PackedAccepted
  split_ifs <;> simp_all

def checkDequant (p : PackedInfo) (dtype : Host.DType) (device : Option Nat) : Except Error Unit :=
  match checkPacked p 1 dtype with
  | .error e => .error e
  | .ok _ => if device=some p.weight.tensor.device then .ok () else .error .device

theorem checkDequant_iff (p : PackedInfo) (dtype : Host.DType) (device : Option Nat) :
    checkDequant p dtype device=.ok () ↔ PackedAccepted p 1 dtype ∧ device=some p.weight.tensor.device := by
  unfold checkDequant
  cases hp : checkPacked p 1 dtype <;> simp [hp,←checkPacked_iff]

structure S1Info where
  packedInfo : PackedInfo
  input : Info
  supplied : Option Info

def arguments (p : S1Info) : Host.Arguments :=
  ⟨packed p.packedInfo p.input.tensor.rows,p.input.tensor,p.supplied.map Info.tensor⟩

def S1Accepted (p : S1Info) : Prop :=
  PackedAccepted p.packedInfo p.input.tensor.rows .fp16 ∧ Host.Contract (arguments p) ∧
  Ordinary p.input .fp16 (arguments p).packed.shape.m (arguments p).packed.shape.k p.packedInfo.weight.tensor.device ∧
  ∀ out,p.supplied=some out → Ordinary out .fp16 (arguments p).packed.shape.m
    (arguments p).packed.shape.n p.packedInfo.weight.tensor.device

def checkS1 (p : S1Info) : Except Error Unit :=
  if S1Accepted p then .ok () else .error .geometry
theorem checkS1_iff (p : S1Info) : checkS1 p=.ok () ↔ S1Accepted p := by simp [checkS1]

structure QuantInfo where
  input : Info
  group : Nat
  groupIsInt : Bool
  rank : Nat
  rankIsInt : Bool
def quantShape (p : QuantInfo) : Shape := ⟨1,p.input.tensor.columns,p.input.tensor.rows,p.group⟩
def QuantMetadata (p : QuantInfo) (dtype : Host.DType) : Prop :=
  p.groupIsInt=true ∧ p.rankIsInt=true ∧ (quantShape p).Valid ∧
  Ordinary p.input dtype (quantShape p).k (quantShape p).n p.input.tensor.device ∧
  p.rank=p.input.tensor.device ∧ (quantShape p).k*(quantShape p).n<2^31

/-- The content guards inspect original bytes and the actual computed scales.
The two division operation families remain distinct inside Native.quant*. -/
def checkQuant16 (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (bytes : Bytes) : Except Error Unit :=
  if ¬QuantMetadata p .fp16 then .error .geometry
  else if ¬(∀ k n,k<(quantShape p).k → n<(quantShape p).n →
    finite16 ((load halfCodec bytes).values .weight (k*(quantShape p).n+n))) then .error .finite
  else if ¬(∀ g n,g<(quantShape p).k/(quantShape p).group → n<(quantShape p).n →
    positive16 (scale (quantShape p) (Native.quant16 I) order (load halfCodec bytes) g n)) then .error .scale
  else .ok ()
def checkQuant32 (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (bytes : Bytes) : Except Error Unit :=
  if ¬QuantMetadata p .fp32 then .error .geometry
  else if ¬(∀ k n,k<(quantShape p).k → n<(quantShape p).n →
    finite32 ((load wordCodec bytes).values .weight (k*(quantShape p).n+n))) then .error .finite
  else if ¬(∀ g n,g<(quantShape p).k/(quantShape p).group → n<(quantShape p).n →
    positive32 (scale (quantShape p) (Native.quant32 I) order (load wordCodec bytes) g n)) then .error .scale
  else .ok ()

theorem quant16_checked (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (bytes : Bytes) (checked : checkQuant16 p I order bytes=.ok ()) :
    QuantMetadata p .fp16 ∧ QuantDomain16 (quantShape p) I order bytes p.input.tensor p.rank := by
  unfold checkQuant16 at checked
  split_ifs at checked with metadata finite positive <;> simp_all
  exact ⟨metadata.2.2.1,metadata.2.2.2.1.2.2.2.2.2.2.1,metadata.2.2.2.2.1,
    metadata.2.2.2.2.2,finite,positive⟩
theorem quant32_checked (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (bytes : Bytes) (checked : checkQuant32 p I order bytes=.ok ()) :
    QuantMetadata p .fp32 ∧ QuantDomain32 (quantShape p) I order bytes p.input.tensor p.rank := by
  unfold checkQuant32 at checked
  split_ifs at checked with metadata finite positive <;> simp_all
  exact ⟨metadata.2.2.1,metadata.2.2.2.1.2.2.2.2.2.2.1,metadata.2.2.2.2.1,
    metadata.2.2.2.2.2,finite,positive⟩

theorem formed_pointer (i : Info) (dtype : Host.DType) (rows columns device : Nat)
    (h : Ordinary i dtype rows columns device) (off : Nat) (bound : off<2^31) :
    i.tensor.base+dtype.width*off<2^64 := by
  have width : dtype.width≤4 := by cases dtype <;> decide
  have ptr := h.2.2.2.2.2.2.2.2
  nlinarith

#axiomsClean checkPacked_iff
#axiomsClean checkS1_iff
#axiomsClean quant16_checked
#axiomsClean quant32_checked
#axiomsClean formed_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
