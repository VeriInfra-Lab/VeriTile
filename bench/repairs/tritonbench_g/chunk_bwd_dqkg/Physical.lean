import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Host
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.ByteFrame
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Domain

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
open VeriTile.Triton Whole
noncomputable section

structure PhysicalTensor where
  descriptor : Tensor
  byteBase : Nat
  cellWidth : Nat

def placed (A : FlatAlloc) (t : Tensor) : PhysicalTensor := ⟨t,4*A.base t.region,4⟩
def physicalDescriptors (c : Config) (device : Nat) (A : FlatAlloc) :
    PhysicalTensor × PhysicalTensor × PhysicalTensor :=
  let t:=outputDescriptors c device
  (placed A t.1,placed A t.2.1,placed A t.2.2)

structure PhysicalResult where
  state : BlockState
  tensors : PhysicalTensor × PhysicalTensor × PhysicalTensor

def invokePhysical (c : Config) (device : Nat) (A : FlatAlloc) (before : BlockState)
    (alloc : Allocation before) (order : List (CTA c)) : Option PhysicalResult :=
  (runFlat c A order (A.flattenState alloc.state)).map fun state =>
    ⟨state,physicalDescriptors c device A⟩

/-- Public physical whole-call endpoint. Returned descriptor bases address the
same allocation used by the executed source. Every logical output, full cell
frame and full byte frame are composed here; no successful run is a premise. -/
theorem correct_physical (c : Config) (device : Nat) (A : FlatAlloc)
    (ha : DenseAllocation c A) (before : BlockState) (alloc : Allocation before)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (encode : MemCell → BitVec 32) (raw : Nat → BitVec 8)
    (hb : ByteBinding A encode raw (A.flattenState alloc.state)) :
    ∃ result,invokePhysical c device A before alloc order=some result ∧
      result.tensors=physicalDescriptors c device A ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
        result.state.readMem A.flat (result.tensors.1.byteBase/4+(h.val*c.T+t.val)*c.K+e.val)=
          Spec.dq c (Spec.views c before.readMem) h.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
        result.state.readMem A.flat (result.tensors.2.1.byteBase/4+(h.val*c.T+t.val)*c.K+e.val)=
          Spec.dk c (Spec.views c before.readMem) h.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T),
        result.state.readMem A.flat (result.tensors.2.2.byteBase/4+(k.val*c.heads+h.val)*c.T+t.val)=
          Spec.dg c (Spec.views c before.readMem) k.val h.val (t.val/64) (t.val%64)) ∧
      BlockState.WriteWithin (flatWrites c A) (A.flattenState alloc.state) result.state ∧
      (∀ i,¬byteWrites c A i → materializeBytes A encode raw result.state i=raw i) := by
  obtain ⟨final,he,hq,hk,hg,hf⟩ := whole_flat_result c A ha alloc.state order ho
  have hi := allocation_inputs before alloc
  refine ⟨⟨final,physicalDescriptors c device A⟩,by simp [invokePhysical,he],rfl,?_,?_,?_,hf,?_⟩
  · intro h t e
    have value : final.readMem A.flat (A.addr "dq" ((h.val*c.T+t.val)*c.K+e.val)) =
        expectedQ c before h t e := by simpa only [expectedQ,hi] using hq h t e
    rw [Spec.logical_q] at value
    simpa [physicalDescriptors,outputDescriptors,placed,tensor,FlatAlloc.addr,Nat.add_assoc] using value
  · intro h t e
    have value : final.readMem A.flat (A.addr "dk" ((h.val*c.T+t.val)*c.K+e.val)) =
        expectedK c before h t e := by simpa only [expectedK,hi] using hk h t e
    rw [Spec.logical_k] at value
    simpa [physicalDescriptors,outputDescriptors,placed,tensor,FlatAlloc.addr,Nat.add_assoc] using value
  · intro k h t
    have value : final.readMem A.flat (A.addr "dg" ((k.val*c.heads+h.val)*c.T+t.val)) =
        expectedG c before k h t := by simpa only [expectedG,hi] using hg k h t
    rw [Spec.logical_g] at value
    simpa [physicalDescriptors,outputDescriptors,placed,tensor,FlatAlloc.addr,Nat.add_assoc] using value
  · exact fun i hn => byte_frame c A encode raw (A.flattenState alloc.state) final hb hf i hn

#axiomsClean correct_physical
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg
