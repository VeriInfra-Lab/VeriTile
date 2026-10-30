import bench.repairs.tritonbench_g.chunk_retention.Proofs.NativePrimitives
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ByteFrames

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

structure ForwardReturn where
  output : Tensor
  finalState : Option Tensor
  saved : List Tensor
  launches : List Phase
  launchDevice : Nat
  restoredDevice : Nat
  memory : ByteMemory

def forwardLaunches (c : Shape) : List Phase :=
  if c.heads=0 then [] else [.h]++if c.chunks=0 then [] else [.o]

def forwardMemory (c : Shape) (f : ForwardObjects c) (p : Primitives)
    (tree : SumTree) (before : ByteMemory) : ByteMemory :=
  finishBytes c f.tensor (Separate.forwardFootprint c f.flag)
    (Source.afterO (nativeOps p) c (nativeScale p c) tree f.flag (forwardHeap c f before)) before

def forwardReturn (c : Shape) (f : ForwardObjects c) (p : Primitives)
    (tree : SumTree) (priorDevice : Nat) (before : ByteMemory) : ForwardReturn :=
  ⟨f.output,f.finalState,[f.q,f.k,f.v,f.history],forwardLaunches c,f.q.device,priorDevice,
    forwardMemory c f p tree before⟩

def forwardCall (c : Shape) (f : ForwardObjects c) (p : Primitives)
    (tree : SumTree) (priorDevice : Nat) (before : ByteMemory) : Option ForwardReturn :=
  if forwardChecks c f then some (forwardReturn c f p tree priorDevice before) else none

/-- Independent reference readbacks from actual returned object pointers. -/
def ForwardValues (c : Shape) (f : ForwardObjects c) (p : Primitives)
    (before after : ByteMemory) : Prop :=
  (∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    readWord after (f.output.dataPointer+4*c.valueOffset h t j)=
      Reference.output (nativeOps p) c (inputsOfHeap (forwardHeap c f before)) (nativeScale p c) h t j) ∧
  (∀ final,f.finalState=some final → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    readWord after (final.dataPointer+4*c.stateOffset h e j)=
      Reference.finalState (nativeOps p) c (inputsOfHeap (forwardHeap c f before)) h e j)

theorem forward_values (c : Shape) (f : ForwardObjects c) (p : Primitives) (tree : SumTree)
    (before : ByteMemory) (checks : forwardChecks c f=Bool.true)
    (fresh : SeparateStores f.tensor (Separate.forwardFootprint c f.flag)) :
    ForwardValues c f p before (forwardMemory c f p tree before) := by
  have admit:=forward_store_admission c f checks
  have source:=Separate.forward_post (nativeOps p) c (nativeScale p c) tree f.flag (forwardHeap c f before)
  constructor
  · intro h hh t ht j hj
    have hm : (⟨.output,h,0,t,j⟩:Address)∈Separate.forwardFootprint c f.flag :=
      List.mem_append_right _ (Source.output_cover c h t j hh ht hj)
    exact (finish_word c f.tensor _ admit fresh _ before _ hm).trans (source.output h hh t ht j hj)
  · intro final hf h hh e he j hj
    have flag : f.flag=Bool.true := by rw [←f.finalFlag,hf];rfl
    have hm : (⟨.finalState,h,0,e,j⟩:Address)∈Separate.forwardFootprint c f.flag := by
      rw [flag];exact List.mem_append_left _ (Source.final_cover c h e j hh he hj)
    have hw:=finish_word c f.tensor _ admit fresh
      (Source.afterO (nativeOps p) c (nativeScale p c) tree f.flag (forwardHeap c f before)) before _ hm
    have hv:=hw.trans (source.finalState flag h hh e he j hj)
    simpa only [cellAddress,ForwardObjects.tensor,hf,Option.getD_some,Address.offset] using hv

/-- No future cotangent or backward allocation is a forward premise. Ordinary
contiguous-copy interpretation supplies the normalized caller objects; the
checked source itself determines all written words and four-byte stores. -/
theorem forward_native (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (actual reference : Primitives) (tree : SumTree)
    (priorDevice : Nat) (before : ByteMemory) (caller : List Tensor)
    (checks : forwardChecks (checkedShape g hs) f=Bool.true)
    (allocation : ForwardAllocation (checkedShape g hs) f caller) (numeric : LocalLaws actual reference) :
    ∃ result,forwardCall (checkedShape g hs) f actual tree priorDevice before=some result ∧
      result.output=f.output ∧ result.finalState=f.finalState ∧ result.finalState.isSome=f.flag ∧
      result.saved=[f.q,f.k,f.v,f.history] ∧ result.launches=forwardLaunches (checkedShape g hs) ∧
      result.launchDevice=f.q.device ∧ result.restoredDevice=priorDevice ∧
      HostABI.Accepted (checkedShape g hs) ∧ ForwardValues (checkedShape g hs) f reference before result.memory ∧
      (∀ a∈Separate.forwardFootprint (checkedShape g hs) f.flag,∀ j : Fin 4,
        extendMemory result.memory (byteAddress (checkedShape g hs) f.tensor (a,j))=
          encode (Source.afterO (nativeOps actual) (checkedShape g hs) (nativeScale actual (checkedShape g hs)) tree f.flag
            (forwardHeap (checkedShape g hs) f before) a) j) ∧
      (∀ point : Fin (2^64),
        (∀ i∈byteEvents (Separate.forwardFootprint (checkedShape g hs) f.flag),
          point.val≠byteAddress (checkedShape g hs) f.tensor i) → result.memory point=before point) ∧
      (∀ t∈caller,∀ point : Fin (2^64),t.storageBase≤point.val →
        point.val<t.storageBase+t.storageBytes → result.memory point=before point) := by
  have admit:=forward_store_admission (checkedShape g hs) f checks
  have hp : planChecks (checkedShape g hs)=Bool.true := by
    simp only [forwardChecks,Bool.and_eq_true] at checks
    exact checks.1.1
  refine ⟨forwardReturn _ f actual tree priorDevice before,?_,rfl,rfl,f.finalFlag,rfl,rfl,rfl,rfl,
    plan_admitted g hs hp,?_,?_,?_,?_⟩
  · simp [forwardCall,checks]
  · rw [←local_eq actual reference numeric]
    exact forward_values _ f actual tree before checks allocation.writes
  · intro a ha j
    exact finish_byte _ f.tensor _ admit allocation.writes _ before a ha j
  · intro point outside
    exact finish_frame _ _ _ _ _ point outside
  · intro t ht point hlo hhi
    exact finish_preserves_allocation _ f.tensor _ admit _ before t
      (fun a ha=>allocation.caller a ha t ht) point hlo hhi

#axiomsClean forward_values
#axiomsClean forward_native
#specNonCircular ForwardValues avoiding [forwardCall,forwardMemory,forward_native]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
