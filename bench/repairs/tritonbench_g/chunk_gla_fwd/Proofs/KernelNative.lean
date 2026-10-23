import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelTensor
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelRuntime

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.KernelNative
open VeriTile Triton KernelTensor NativeInstructions
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def inputTensors (r : Admission.Request) : List (RegionName × Admission.Tensor) :=
  match r.mode with
  | .producer => [("q",r.query),("k",r.key),("g",r.gate)]
  | .consumer => [("q",r.query),("v",r.value),("g",r.gate),("attention",r.attention),("h",r.history)]

/-- Ordinary data_ptr-to-word-address observations. This is metadata only. -/
def InputBinding (r : Admission.Request) (abi : Physical.ABI c) : Prop :=
  ∀ pair ∈ inputTensors r,pair.2.byteBase=4*(abi.alloc pair.1).base

def execute (I : ISA (BitVec 32)) (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (before : Nat → BitVec 32) : Option (Nat → BitVec 32) :=
  (KernelRuntime.execute mode w scale abi before).map (fun m p => (m p).eval I)

def target (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (before : Nat → BitVec 32) : Index mode c → Graph :=
  match mode with
  | .producer => fun i => KernelRuntime.producerValue w scale abi before
      (head c i.1 i.2.1) i.2.2.1 i.2.2.2
  | .consumer => fun i => KernelRuntime.consumerValue w scale abi before
      (head c i.1 i.2.1) i.2.2.1 i.2.2.2

/-- Includes the unused scratch allocation in the preserved frame whenever
the direct branch or an empty launch is selected. -/
def Outside (mode : Admission.Mode) (abi : Physical.ABI c) (p : Nat) : Prop :=
  match mode with
  | .producer => if c.heads*c.T=0 ∨ c.K≤256 then ¬(abi.alloc "attention").Contains p
    else ¬(abi.alloc "attention").Contains p ∧ ¬(abi.alloc "scratch").Contains p
  | .consumer => ¬(abi.alloc "o").Contains p

theorem graph_frame (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (before : Nat → BitVec 32) (p : Nat) (hp : Outside mode abi p) :
    KernelRuntime.reference mode w scale abi before p=.bits (before p) := by
  cases mode with
  | consumer => exact KernelRuntime.frame .consumer w scale abi before p hp
  | producer =>
    change NativeStaging.lower (Physical.producerReference (KernelRuntime.algebra w scale)
      NativeStaging.dot abi KernelRuntime.scaleTag (NativeSource.inputMemory before) p)=_
    unfold Physical.producerReference Physical.producerCommit
    split
    · rename_i small
      have outside : ¬(abi.alloc "attention").Contains p := by simpa only [Outside,if_pos small] using hp
      rw [Physical.commit_frame _ _ _ _ _ outside]
      rfl
    · rename_i large
      have outside : ¬(abi.alloc "attention").Contains p ∧ ¬(abi.alloc "scratch").Contains p := by
        simpa only [Outside,if_neg large] using hp
      rw [Physical.commit_frame _ _ _ _ _ outside.1,Physical.commit_frame _ _ _ _ _ outside.2]
      rfl

def Post (I : ISA (BitVec 32)) (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (view : Output mode c device) (before final : Nat → BitVec 32) : Prop :=
  (∀ i,(nativeResult view final).get i=(target mode w scale abi before i).eval I) ∧
  (∀ p,Outside mode abi p → final p=before p) ∧
  (∀ r,Physical.tracked r → KernelRuntime.ReadOnly mode r → ∀ i,i<(abi.alloc r).length →
    final ((abi.alloc r).base+i)=before ((abi.alloc r).base+i))

/-- Raw physical output words and the unchanged exterior/input-region frame. -/
def RawPost (I : ISA (BitVec 32)) (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (before final : Nat → BitVec 32) : Prop :=
  (∀ i,final ((abi.alloc (region mode)).base+offset mode c i)=(target mode w scale abi before i).eval I) ∧
  (∀ p,Outside mode abi p → final p=before p) ∧
  (∀ r,Physical.tracked r → KernelRuntime.ReadOnly mode r → ∀ i,i<(abi.alloc r).length →
    final ((abi.alloc r).base+i)=before ((abi.alloc r).base+i))

/-- Entire shared-source execution, independent instruction reference, actual
Tensor coordinates and complete relevant physical frame. The only numerical
parameter is the local scalar/fixed-instruction ISA. Raw input bit leaves are
identities; no program, grid, reduction, output or frame equality is assumed. -/
theorem raw_complete (I : ISA (BitVec 32)) (bits : ∀ b,I.bits b=b)
    (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (fresh : KernelRuntime.Fresh mode abi)
    (before : Nat → BitVec 32) :
    ∃ final,execute I mode w scale abi before=some final ∧ RawPost I mode w scale abi before final := by
  let final := fun p => (KernelRuntime.reference mode w scale abi before p).eval I
  refine ⟨final,?_,?_,?_,?_⟩
  · simp [execute,KernelRuntime.source_complete,final]
  · intro i
    cases mode with
    | producer =>
      change (KernelRuntime.reference .producer w scale abi before
        ((abi.alloc "attention").base+offset .producer c i)).eval I=_
      exact congrArg (Graph.eval I)
        (KernelRuntime.producer_readback w scale abi before (head c i.1 i.2.1) i.2.2.1 i.2.2.2)
    | consumer =>
      change (KernelRuntime.reference .consumer w scale abi before
        ((abi.alloc "o").base+offset .consumer c i)).eval I=_
      have value := KernelRuntime.consumer_readback w scale abi before
        (head c i.1 i.2.1) i.2.2.1 i.2.2.2
      simpa only [offset,width,target,Nat.mul_assoc,Nat.add_assoc] using congrArg (Graph.eval I) value
  · intro p hp
    change (KernelRuntime.reference mode w scale abi before p).eval I=before p
    rw [graph_frame mode w scale abi before p hp]
    exact bits _
  · intro r hr ro i hi
    change (KernelRuntime.reference mode w scale abi before ((abi.alloc r).base+i)).eval I=_
    rw [KernelRuntime.inputs mode w scale abi fresh before r hr ro i hi]
    exact bits _


theorem complete (I : ISA (BitVec 32)) (bits : ∀ b,I.bits b=b)
    (mode : Admission.Mode) (w : KernelProfile.Profile) (scale : BitVec 32)
    (abi : Physical.ABI c) (fresh : KernelRuntime.Fresh mode abi)
    (view : Output mode c device) (base : view.info.byteBase=4*(abi.alloc (region mode)).base)
    (before : Nat → BitVec 32) :
    ∃ final,execute I mode w scale abi before=some final ∧ Post I mode w scale abi view before final := by
  obtain ⟨final,ran,values,frame,inputs⟩ := raw_complete I bits mode w scale abi fresh before
  have base' : view.info.byteBase/4=(abi.alloc (region mode)).base := by rw [base]; omega
  refine ⟨final,ran,?_,frame,inputs⟩
  simpa only [nativeResult,Tensor.get,coordinates,base'] using values

#axiomsClean complete
#axiomsClean graph_frame
#specNonCircular target avoiding [execute,KernelRuntime.execute,FP.Structural.exec,
  Physical.producerExecute,Physical.consumerExecute]
#specNonCircular Post avoiding [execute,KernelRuntime.execute,FP.Structural.exec,
  Physical.producerExecute,Physical.consumerExecute]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.KernelNative
