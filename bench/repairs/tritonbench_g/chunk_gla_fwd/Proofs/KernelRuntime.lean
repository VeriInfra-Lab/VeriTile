import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.KernelRuntime
open VeriTile Triton FP.Structural
open NativeStaging NativeInstructions
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

abbrev scaleTag := NativeRuntime.scaleTag
abbrev algebra := KernelProfile.algebra

def execute (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (m : Nat → BitVec 32) : Option (Nat → Graph) :=
  ((match mode with
    | .producer => Physical.producerExecute (algebra w scale) dot abi scaleTag (NativeSource.inputMemory m)
    | .consumer => Physical.consumerExecute (algebra w scale) dot abi scaleTag (NativeSource.inputMemory m))).map
      NativeSource.materialize

def reference (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (m : Nat → BitVec 32) : Nat → Graph :=
  NativeSource.materialize (match mode with
    | .producer => Physical.producerReference (algebra w scale) dot abi scaleTag (NativeSource.inputMemory m)
    | .consumer => Physical.consumerReference (algebra w scale) dot abi scaleTag (NativeSource.inputMemory m))

theorem source_complete (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (m : Nat → BitVec 32) :
    execute mode w scale abi m=some (reference mode w scale abi m) := by
  cases mode <;> simp [execute,reference,Physical.producer_source,Physical.consumer_source]

def producerValue (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c) (m : Nat → BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) : Graph :=
  lower (Host.producerReference (algebra w scale) dot c scaleTag
    (FPStorage.project abi.alloc (NativeSource.inputMemory m)) "attention"
      (h.val*c.T*c.BT+t.val*c.BT+j.val))

def consumerValue (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c) (m : Nat → BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : Graph :=
  lower (FPReturned.value (algebra w scale) dot c.consumer scaleTag
    (FPStorage.project abi.alloc (NativeSource.inputMemory m)) h t v)

theorem producer_readback (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c) (m : Nat → BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    reference .producer w scale abi m ((abi.alloc "attention").base+(h.val*c.T*c.BT+t.val*c.BT+j.val))=
      producerValue w scale abi m h t j := by
  apply congrArg lower
  apply Physical.producer_readback
  simpa only [Host.attentionSize,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound h.isLt t.isLt j.isLt

theorem consumer_readback (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c) (m : Nat → BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    reference .consumer w scale abi m ((abi.alloc "o").base+h.val*(c.T*c.V)+t.val*c.V+v.val)=
      consumerValue w scale abi m h t v :=
  congrArg lower (Physical.consumer_tensor (algebra w scale) dot abi scaleTag (NativeSource.inputMemory m) h t v)

def Fresh (mode : Admission.Mode) (abi : Physical.ABI c) : Prop :=
  match mode with
  | .producer => Physical.ProducerFresh abi
  | .consumer => Physical.ConsumerFresh abi

def Outside (mode : Admission.Mode) (abi : Physical.ABI c) (p : Nat) : Prop :=
  match mode with
  | .producer => ¬(abi.alloc "attention").Contains p ∧ ¬(abi.alloc "scratch").Contains p
  | .consumer => ¬(abi.alloc "o").Contains p

theorem frame (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (m : Nat → BitVec 32) (p : Nat) (hp : Outside mode abi p) :
    reference mode w scale abi m p=.bits (m p) := by
  cases mode with
  | producer => exact congrArg lower (Physical.producer_frame (algebra w scale) dot abi scaleTag _ p hp.1 hp.2)
  | consumer => exact congrArg lower (Physical.consumer_frame (algebra w scale) dot abi scaleTag _ p hp)

theorem producer_inputs (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c)
    (fresh : Physical.ProducerFresh abi) (m : Nat → BitVec 32)
    (r : RegionName) (hr : Physical.tracked r) (ha : r≠"attention") (hs : r≠"scratch")
    (i : Nat) (hi : i<(abi.alloc r).length) :
    reference .producer w scale abi m ((abi.alloc r).base+i)=.bits (m ((abi.alloc r).base+i)) :=
  congrArg lower (Physical.producer_preserves_input (algebra w scale) dot abi fresh scaleTag _ r hr ha hs i hi)

theorem consumer_inputs (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c)
    (fresh : Physical.ConsumerFresh abi) (m : Nat → BitVec 32)
    (r : RegionName) (hr : Physical.tracked r) (ho : r≠"o") (i : Nat) (hi : i<(abi.alloc r).length) :
    reference .consumer w scale abi m ((abi.alloc r).base+i)=.bits (m ((abi.alloc r).base+i)) :=
  congrArg lower (Physical.consumer_preserves_input (algebra w scale) dot abi fresh scaleTag _ r hr ho i hi)

def ReadOnly (mode : Admission.Mode) (r : RegionName) : Prop :=
  match mode with
  | .producer => r≠"attention" ∧ r≠"scratch"
  | .consumer => r≠"o"

theorem inputs (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32) (abi : Physical.ABI c)
    (fresh : Fresh mode abi) (m : Nat → BitVec 32) (r : RegionName)
    (hr : Physical.tracked r) (ro : ReadOnly mode r) (i : Nat) (hi : i<(abi.alloc r).length) :
    reference mode w scale abi m ((abi.alloc r).base+i)=.bits (m ((abi.alloc r).base+i)) := by
  cases mode with
  | producer => exact producer_inputs w scale abi fresh m r hr ro.1 ro.2 i hi
  | consumer => exact consumer_inputs w scale abi fresh m r hr ro i hi

theorem transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (ports : LocalTransport f A B) (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (m : Nat → BitVec 32) (p : Nat) :
    f ((reference mode w scale abi m p).eval A)=(reference mode w scale abi m p).eval B :=
  NativeInstructions.transport f A B ports _

#specNonCircular producerValue avoiding [execute,FP.Structural.exec,Physical.producerExecute]
#specNonCircular consumerValue avoiding [execute,FP.Structural.exec,Physical.consumerExecute]
#axiomsClean source_complete
#axiomsClean producer_readback
#axiomsClean consumer_readback
#axiomsClean frame
#axiomsClean transport
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.KernelRuntime
