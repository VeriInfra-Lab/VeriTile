import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Contract
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.RealForward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.PhysicalLayout
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Host
open Admission
noncomputable section

/-- Ordinary tensor primitive results. Object identities are supplied by
new_empty/new_zeros, squeeze, sum and same-dtype/device to, not invented by a
numerical postcondition. Source regions index the actual backing allocations. -/
structure Objects where
  region : Region → Descriptor
  squeezedOutput : Nat

def observations (inputs : List Region) (objects : Objects) : List (Region×Descriptor) :=
  inputs.map (fun r => (r,objects.region r))

def cotangentCheck (c : ABI.Call) (inputs : List Region) : Bool :=
  decide (Region.do_∈inputs ∧ (c.terminalCotangent=Bool.true → Region.dht∈inputs))

structure TensorObject where
  identity : Nat
  storage : Nat
  base : Nat
  device : Nat
  shape : List Nat

def tensor (c : ABI.Call) (objects : Objects) (r : Region) : TensorObject :=
  ⟨(objects.region r).object,(objects.region r).storage,(objects.region r).base,
    (objects.region r).device,(ABI.tensorMeta c.shape (ABI.flags c) r).shape⟩

def outputObject (c : ABI.Call) (objects : Objects) : TensorObject :=
  {tensor c objects .o with identity := objects.squeezedOutput}

def outputTuple (c : ABI.Call) (objects : Objects) : TensorObject×Option TensorObject :=
  (outputObject c objects,if c.outputFinal then some (tensor c objects .ht) else none)

def gradientTuple (c : ABI.Call) (objects : Objects) : List (Option TensorObject) :=
  [some (tensor c objects .retq),some (tensor c objects .retk),some (tensor c objects .retv),
   some (tensor c objects .retbeta),none,
   if c.initial && c.initialRequiresGrad then some (tensor c objects .dh0) else none,none]

structure Saved where
  inputs : List Nat
  scale : ABI.ScaleArgument
  outputFinal : Bool
  used : Bool

def saved (c : ABI.Call) (objects : Objects) : Saved :=
  ⟨[.q,.k,.v,.beta].map (fun r=>(objects.region r).object) ++
    (if c.initial then [(objects.region .h0).object] else []),c.scale,c.outputFinal,Bool.false⟩

structure Return (M : Type) (A : Type) where
  memory : M
  value : Except Unit A
  context : Option Saved

def allocationCheck (c : ABI.Call) (objects : Objects) (inputs regions : List Region) : Bool :=
  regions.all (fun r => windowCheck c r (objects.region r)) &&
    separationCheck c (observations (inputs++regions) objects)

def forwardAllocations (c : ABI.Call) : List Region := [.o]++(if c.outputFinal then [.ht] else [])
def backwardAllocations (c : ABI.Call) : List Region :=
  [.dq,.dk,.dv,.dbeta]++(if c.initial && c.initialRequiresGrad then [.dh0] else [])

/-- Guard rejection and allocation-window rejection precede source execution.
The `device` parameter is q.device inside each standard owning-device context;
the context restores the ambient caller device after submission.
Allocation failures have no completed return and are outside these successful
ordinary primitive observations. The source itself computes every stored byte. -/
def forwardFP (c : ABI.Call) (device : Nat) (inputs : List Region)
    (outputIsBool scaleAccepted : Bool) (objects : Objects) (g : Trees)
    (actual : Native.Instructions) (policy : Native.Policy) (before : PhysicalLayout.Bytes) :
    Return PhysicalLayout.Bytes (TensorObject×Option TensorObject) :=
  if check c device (observations inputs objects) outputIsBool scaleAccepted && allocationCheck c objects inputs (forwardAllocations c) then
    ⟨PhysicalLayout.forward c.shape (ABI.flags c) g c.scale.launch actual policy before,
      .ok (outputTuple c objects),some (saved c objects)⟩
  else ⟨before,.error (),none⟩

def backwardFP (c : ABI.Call) (device : Nat) (inputs : List Region)
    (outputIsBool scaleAccepted higherOrder : Bool) (context : Saved) (objects : Objects) (g : Trees)
    (actual : Native.Instructions) (policy : Native.Policy) (before : PhysicalLayout.Bytes) :
    Return PhysicalLayout.Bytes (List (Option TensorObject)) :=
  if check c device (observations inputs objects) outputIsBool scaleAccepted && backwardCheck context.used higherOrder && cotangentCheck c inputs &&
      allocationCheck c objects inputs (backwardAllocations c) then
    ⟨PhysicalLayout.backward c.shape (ABI.flags c) g c.scale.launch actual policy before,
      .ok (gradientTuple c objects),some {context with used:=Bool.true}⟩
  else ⟨before,.error (),some context⟩

def forwardReal (c : ABI.Call) (device : Nat) (inputs : List Region)
    (outputIsBool scaleAccepted : Bool) (objects : Objects) (g : Trees) (before : Memory ℝ) :
    Return (Memory ℝ) (TensorObject×Option TensorObject) :=
  if check c device (observations inputs objects) outputIsBool scaleAccepted && allocationCheck c objects inputs (forwardAllocations c) then
    ⟨forwardCall c.shape (ABI.flags c) Real.ops g (ABI.realScale c) before,
      .ok (outputTuple c objects),some (saved c objects)⟩
  else ⟨before,.error (),none⟩

def backwardReal (c : ABI.Call) (device : Nat) (inputs : List Region)
    (outputIsBool scaleAccepted higherOrder : Bool) (context : Saved) (objects : Objects) (g : Trees)
    (before : Memory ℝ) : Return (Memory ℝ) (List (Option TensorObject)) :=
  if check c device (observations inputs objects) outputIsBool scaleAccepted && backwardCheck context.used higherOrder && cotangentCheck c inputs &&
      allocationCheck c objects inputs (backwardAllocations c) then
    ⟨backwardCall c.shape (ABI.flags c) Real.ops g (ABI.realScale c) before,
      .ok (gradientTuple c objects),some {context with used:=Bool.true}⟩
  else ⟨before,.error (),some context⟩

theorem forward_rejected (bad : check c device (observations inputs objects) outputIsBool scaleAccepted=Bool.false) :
    (forwardFP c device inputs outputIsBool scaleAccepted objects g actual policy before).memory=before ∧
    (forwardFP c device inputs outputIsBool scaleAccepted objects g actual policy before).value=.error () := by
  simp [forwardFP,bad]

theorem backward_rejected (bad : backwardCheck context.used higherOrder=Bool.false) :
    (backwardFP c device inputs outputIsBool scaleAccepted higherOrder context objects g actual policy before).memory=before ∧
    (backwardFP c device inputs outputIsBool scaleAccepted higherOrder context objects g actual policy before).value=.error () := by
  simp [backwardFP,bad]

theorem forward_transport (ports : Native.Obligations actual) :
    forwardFP c device inputs outputIsBool scaleAccepted objects g actual policy before=
      forwardFP c device inputs outputIsBool scaleAccepted objects g Native.software policy before := by
  unfold forwardFP
  split_ifs
  · rw [PhysicalLayout.forward_transport c.shape (ABI.flags c) g c.scale.launch actual ports policy]
  · rfl

theorem backward_transport (ports : Native.Obligations actual) :
    backwardFP c device inputs outputIsBool scaleAccepted higherOrder context objects g actual policy before=
      backwardFP c device inputs outputIsBool scaleAccepted higherOrder context objects g Native.software policy before := by
  unfold backwardFP
  split_ifs
  · rw [PhysicalLayout.backward_transport c.shape (ABI.flags c) g c.scale.launch actual ports policy]
  · rfl

theorem forward_frame (r : Region) (byte : Nat)
    (outside : ∀ loc,PhysicalLayout.At c.shape (ABI.flags c) r byte loc →
      ¬MemoryProof.ForwardCallWrites c.shape (ABI.flags c) loc.1) :
    (forwardFP c device inputs outputIsBool scaleAccepted objects g actual policy before).memory r byte=before r byte := by
  unfold forwardFP
  split_ifs
  · exact PhysicalLayout.publish_frame c.shape (ABI.flags c) _ before _ r byte outside
  · rfl

theorem backward_frame (r : Region) (byte : Nat)
    (outside : ∀ loc,PhysicalLayout.At c.shape (ABI.flags c) r byte loc →
      ¬MemoryProof.BackwardWrites c.shape (ABI.flags c) loc.1) :
    (backwardFP c device inputs outputIsBool scaleAccepted higherOrder context objects g actual policy before).memory r byte=before r byte := by
  unfold backwardFP
  split_ifs
  · exact PhysicalLayout.publish_frame c.shape (ABI.flags c) _ before _ r byte outside
  · rfl

abbrev Heap := Nat → Codec.Byte

/-- Actual pointer addresses. Source writes are installed at the admitted
caller/allocation bases; every other byte of the global heap is retained. -/
def AtByte (c : ABI.Call) (objects : Objects) (address : Nat) (loc : Addr×Fin 4) : Prop :=
  Geometry.Valid c.shape (ABI.flags c) loc.1 ∧
    address=(objects.region loc.1.region).base+4*Geometry.offset c.shape (ABI.flags c) loc.1+loc.2.val

def heap (c : ABI.Call) (objects : Objects) (writes : Addr → Prop)
    (before : Heap) (bytes : PhysicalLayout.Bytes) : Heap := by
  classical
  exact fun address => if h : ∃ loc,AtByte c objects address loc ∧ writes loc.1 then
    bytes h.choose.1.region (4*Geometry.offset c.shape (ABI.flags c) h.choose.1+h.choose.2.val)
    else before address

theorem global_frame (c : ABI.Call) (objects : Objects) (writes : Addr → Prop)
    (before : Heap) (bytes : PhysicalLayout.Bytes) (address : Nat)
    (outside : ∀ loc,AtByte c objects address loc → ¬writes loc.1) :
    heap c objects writes before bytes address=before address := by
  unfold heap
  rw [dif_neg]
  rintro ⟨loc,ha,hw⟩
  exact outside loc ha hw

#axiomsClean forward_rejected
#axiomsClean backward_rejected
#axiomsClean forward_transport
#axiomsClean backward_transport
#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean global_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Host
