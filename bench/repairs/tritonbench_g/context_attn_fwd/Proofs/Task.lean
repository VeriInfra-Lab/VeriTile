import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Tensor
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FPReferenceGrid

namespace VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor
open VeriTile.Triton FP.Structural NativeGraph Returned
open VeriTile.Torch.FP32 (logicalValue)
noncomputable section
local instance (c : Host.Call) : Decidable (Needed c) := Classical.propDecidable _

/-- Independent tile recurrence plus pure masked scatter over the complete host grid.
The reference contains no typed source execution or successful-result selector. -/
def referenceMemory (c : Host.Call) (B : Native.Bytes extent) (before : Physical.Memory) :=
  let plan := NativeSM90.plan c.layout.BM c.layout.D
  (FPReference.grid c (backend plan) (dots.support (scalar plan) c.layout)
    (order c) (Snapshot.initial c B before)).mem

/-- FP32 reference evaluated with the explicit fixed-k8 WGMMA chain and scalar
instruction ports, retaining the selected reduction trees and physical routes. -/
def referenceWord (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (before : Physical.Memory) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) : Word :=
  evaluate ops (NativeSM90.plan c.layout.BM c.layout.D)
    ((referenceMemory c B before c.regions.out (c.offset idx i)).read .real)
    (NativeSM90.storedOwner c.layout.BM c.layout.D i.1.val i.2.1.val)

/-- Same supplied Torch output object, complete typed source grid, exact FP32
logical readback, full untouched-byte frame, and observed host return/device.
Metadata and storage interfaces concern only original inputs and API laws. -/
theorem whole_tensor_fp32 (c : Host.Call) (v : Checked.Views) (A : FlatAlloc)
    (B : Native.Bytes A.extent) (start sequence prompt : Nat → Int)
    (checked : Checked.checkInputs c v A.extent start sequence prompt=Bool.true)
    (before : Physical.Memory)
    (observed : Checked.MetadataObservation c B before start sequence prompt)
    (storage : Checked.StorageBinding c v B) (roles : Snapshot.LogicalRoles c)
    (labels : FPSuccess.Separate c) (ops : ScalarOps) (out : Output v) (callerDevice : Nat) :
    let ctx : Context := ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
    ∃ result,invoke c B ops ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=out.tensor.object ∧
      result.currentDevice=callerDevice ∧
      result.launchDevice=(if Needed c then some v.q.metadata.device.index else none) ∧
      FPSuccess.runGrid c (backend (NativeSM90.plan c.layout.BM c.layout.D)) (order c)
        (Snapshot.initial c B before)=some result.graph ∧
      (∀ idx i,c.active idx i →
        (logicalValue (read result.bytes) out.tensor out.laws (flatIndex c idx i)).bits=
          referenceWord c B ops before idx i) ∧
      (∀ a,(¬∃ i : Physical.Lane c,Physical.active c i ∧ Physical.address c B i=a) →
        result.bytes a=before a) ∧
      (∀ rg,rg∈Physical.reads c → ∀ off,off<A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (¬Needed c → result.bytes=before) := by
  dsimp only
  have guard := (Checked.checkInputs_iff c v A.extent start sequence prompt).mp checked
  have call := (Checked.whole_fp32_bytes c v A B start sequence prompt checked before
    observed storage roles labels ops ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩).1
  obtain ⟨result,run,ret,obj,device,launch,source,good,metadata,bytes,frame,inputs,graphFrame,empty⟩ := call
  obtain ⟨referenceGraph,referenceRun,referenceEq⟩ := FPReference.whole c
    (backend (NativeSM90.plan c.layout.BM c.layout.D))
    (dots.support (scalar (NativeSM90.plan c.layout.BM c.layout.D)) c.layout)
    (order c) (Snapshot.initial c B before)
  have same : result.graph=referenceGraph := Option.some.inj (source.symm.trans referenceRun)
  have memory : result.graph.mem=referenceMemory c B before := same ▸ referenceEq
  refine ⟨result,run,ret,obj,device,launch,source,?_,frame,inputs,empty⟩
  intro idx i active
  change _=evaluate ops (NativeSM90.plan c.layout.BM c.layout.D)
    ((referenceMemory c B before c.regions.out (c.offset idx i)).read .real)
    (NativeSM90.storedOwner c.layout.BM c.layout.D i.1.val i.2.1.val)
  rw [← memory]
  apply logical_output c v B guard.1 (Checked.geometry c A.extent guard.2.1) storage out
    result.bytes ops result.graph idx i active
  intro b
  exact bytes (idx,i,b) active

#specNonCircular referenceWord avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,kernel,Returned.invoke]
#axiomsClean whole_tensor_fp32
end
end VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor
