import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardReturn

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
set_option maxHeartbeats 3500000

/-- Public forward values, all stores, and complete byte/allocation frames. -/
def ForwardNativePost (c : Shape) (f : ForwardObjects c) (actual reference : Primitives)
    (tree : SumTree) (priorDevice : Nat) (before : ByteMemory) (caller : List Tensor)
    (result : ForwardReturn) : Prop :=
  result.output=f.output ∧ result.finalState=f.finalState ∧ result.finalState.isSome=f.flag ∧
      result.saved=[f.q,f.k,f.v,f.history] ∧ result.launches=forwardLaunches c ∧
      result.launchDevice=f.q.device ∧ result.restoredDevice=priorDevice ∧
      HostABI.Accepted c ∧ ForwardValues c f reference before result.memory ∧
      (∀ a∈Separate.forwardFootprint c f.flag,∀ j : Fin 4,
        extendMemory result.memory (byteAddress c f.tensor (a,j))=
          encode (Source.afterO (nativeOps actual) c (nativeScale actual c) tree f.flag
            (forwardHeap c f before) a) j) ∧
      (∀ point : Fin (2^64),
        (∀ i∈byteEvents (Separate.forwardFootprint c f.flag),
          point.val≠byteAddress c f.tensor i) → result.memory point=before point) ∧
      (∀ t∈caller,∀ point : Fin (2^64),t.storageBase≤point.val →
        point.val<t.storageBase+t.storageBytes → result.memory point=before point)

/-- Public later backward values, detached returns, all stores, and frames. -/
def BackwardNativePost (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (actual reference : Primitives) (tree : SumTree) (priorDevice : Nat)
    (before later : ByteMemory) (bc : List Tensor) (result : BackwardReturn) : Prop :=
  result.dq=b.dq ∧ result.dk=b.dk ∧ result.dv=b.dv ∧ result.seedGradient=none ∧ result.flagGradient=none ∧
      result.launches=backwardLaunches c ∧ result.launchDevice=f.q.device ∧ result.restoredDevice=priorDevice ∧
      BackwardValues c f b reference tree before later result.memory ∧
      (∀ a∈Separate.backwardFootprint c,∀ j : Fin 4,
        extendMemory result.memory (byteAddress c b.tensor (a,j))=
          encode (Separate.backward (nativeOps actual) c (nativeScale actual c) tree
            (backwardHeap c f b later) a) j) ∧
      (∀ point : Fin (2^64),
        (∀ i∈byteEvents (Separate.backwardFootprint c),point.val≠byteAddress c b.tensor i) →
        (b.incomingPresent=Bool.false → ¬(b.incoming.dataPointer≤point.val ∧ point.val<b.incoming.dataPointer+4*b.incoming.numel)) →
        result.memory point=later point) ∧
      (∀ t∈bc,∀ point : Fin (2^64),t.storageBase≤point.val → point.val<t.storageBase+t.storageBytes →
        result.memory point=later point) ∧
      (∀ t∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList,∀ point : Fin (2^64),
        t.storageBase≤point.val → point.val<t.storageBase+t.storageBytes → result.memory point=later point) ∧
      (b.incomingPresent=Bool.true → ∀ point : Fin (2^64),b.incoming.storageBase≤point.val →
        point.val<b.incoming.storageBase+b.incoming.storageBytes → result.memory point=later point) ∧
      (∀ t,b.terminal=some t → ∀ point : Fin (2^64),t.storageBase≤point.val →
        point.val<t.storageBase+t.storageBytes → result.memory point=later point)

/-- Canonical native task endpoint. Forward depends only on its actual input
snapshot and allocations. Every admissible later backward call is quantified
after the forward return, and obtains its own complete byte-level result. -/
theorem whole_native (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (actual reference : Primitives) (tree : SumTree)
    (complete : tree.Complete (checkedShape g hs).keyBlocks)
    (priorDevice : Nat) (before : ByteMemory) (caller : List Tensor)
    (checks : forwardChecks (checkedShape g hs) f=Bool.true)
    (allocation : ForwardAllocation (checkedShape g hs) f caller) (numeric : LocalLaws actual reference) :
    ∃ result,forwardCall (checkedShape g hs) f actual tree priorDevice before=some result ∧
      ForwardNativePost (checkedShape g hs) f actual reference tree priorDevice before caller result ∧
      ∀ (b : BackwardObjects (checkedShape g hs) f) (later : ByteMemory) (backwardDevice : Nat) (bc : List Tensor),
        backwardChecks (checkedShape g hs) f b=Bool.true →
        BackwardAllocation (checkedShape g hs) f b bc →
        SavedBytes (checkedShape g hs) f result.memory later →
        ∃ gradients,backwardCall (checkedShape g hs) f b actual tree backwardDevice later=some gradients ∧
          BackwardNativePost (checkedShape g hs) f b actual reference tree backwardDevice before later bc gradients := by
  have hf:=forward_native g hs f actual reference tree priorDevice before caller checks allocation numeric
  change ∃ result,forwardCall _ f actual tree priorDevice before=some result ∧
    ForwardNativePost _ f actual reference tree priorDevice before caller result at hf
  obtain ⟨result,hcall,hpost⟩:=hf
  refine ⟨result,hcall,hpost,?_⟩
  intro b later backwardDevice bc bchecks balloc saved
  have hr : result=forwardReturn _ f actual tree priorDevice before := by
    apply Option.some.inj
    simpa only [forwardCall,checks,↓reduceIte] using hcall.symm
  rw [hr] at saved
  exact backward_native g hs f b actual reference tree complete backwardDevice before later caller bc
    checks bchecks allocation balloc saved numeric

#axiomsClean whole_native
#specNonCircular ForwardNativePost avoiding [forwardCall,forward_native,whole_native]
#specNonCircular BackwardNativePost avoiding [backwardCall,backward_native,whole_native]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
