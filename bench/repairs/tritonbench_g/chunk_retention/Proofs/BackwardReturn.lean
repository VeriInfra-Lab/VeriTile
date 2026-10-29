import bench.repairs.tritonbench_g.chunk_retention.Proofs.SavedStorage

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3500000

structure BackwardReturn where
  dq : Tensor
  dk : Tensor
  dv : Tensor
  seedGradient : Option Tensor
  flagGradient : Option Tensor
  launches : List Phase
  launchDevice : Nat
  restoredDevice : Nat
  memory : ByteMemory

def backwardLaunches (c : Shape) : List Phase :=
  (if c.heads=0 ∨ c.chunks=0 then [] else [.dh,.qkv])++[.sum]

def backwardMemory (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (later : ByteMemory) : ByteMemory :=
  finishBytes c b.tensor (Separate.backwardFootprint c)
    (Separate.backward (nativeOps p) c (nativeScale p c) tree (backwardHeap c f b later))
    (prepareBackward b later)

def backwardReturn (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (priorDevice : Nat) (later : ByteMemory) : BackwardReturn :=
  ⟨b.dq,b.dk,b.dv,none,none,backwardLaunches c,f.q.device,priorDevice,backwardMemory c f b p tree later⟩

def backwardCall (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (priorDevice : Nat) (later : ByteMemory) : Option BackwardReturn :=
  if backwardChecks c f b then some (backwardReturn c f b p tree priorDevice later) else none

/-- Input-only native references combine the original detached seed with the
actual later cotangents and saved Q/K/V values. No scratch/result oracle appears. -/
def BackwardValues (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (before later after : ByteMemory) : Prop :=
  let xs:=Separate.joined (forwardHeap c f before) (backwardHeap c f b later)
  (∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    readWord after (b.dq.dataPointer+4*c.keyOffset h t e)=Reference.dq (nativeOps p) c xs (nativeScale p c) h t e) ∧
  (∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    readWord after (b.dk.dataPointer+4*c.keyOffset h t e)=Reference.dk (nativeOps p) c xs (nativeScale p c) h t e) ∧
  (∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    readWord after (b.dv.dataPointer+4*c.valueOffset h t j)=Reference.dv (nativeOps p) c xs (nativeScale p c) tree h t j)

theorem backward_values (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (complete : tree.Complete c.keyBlocks) (before later : ByteMemory)
    (checks : backwardChecks c f b=Bool.true) (fresh : SeparateStores b.tensor (Separate.backwardFootprint c))
    (saved : Separate.Saved (nativeOps p) c (nativeScale p c) tree f.flag (forwardHeap c f before) (backwardHeap c f b later)) :
    BackwardValues c f b p tree before later (backwardMemory c f b p tree later) := by
  have admit:=backward_store_admission c f b checks
  have source:=Separate.backward_post (nativeOps p) c (nativeScale p c) tree complete f.flag
    (forwardHeap c f before) (backwardHeap c f b later) saved
  refine ⟨?_,?_,?_⟩
  · intro h hh t ht e he
    have hm : (⟨.dq,h,0,t,e⟩:Address)∈Separate.backwardFootprint c := by
      simp only [Separate.backwardFootprint,List.mem_append]
      exact Or.inl (Or.inr (Source.dq_cover c h t e hh ht he))
    exact (finish_word c b.tensor _ admit fresh _ (prepareBackward b later) _ hm).trans (source.query h hh t ht e he)
  · intro h hh t ht e he
    have hm : (⟨.dk,h,0,t,e⟩:Address)∈Separate.backwardFootprint c := by
      simp only [Separate.backwardFootprint,List.mem_append]
      exact Or.inl (Or.inr (Source.dk_cover c h t e hh ht he))
    exact (finish_word c b.tensor _ admit fresh _ (prepareBackward b later) _ hm).trans (source.key h hh t ht e he)
  · intro h hh t ht j hj
    have hm : (⟨.dv,h,0,t,j⟩:Address)∈Separate.backwardFootprint c :=
      List.mem_append_right _ (Source.sum_cover c h t j hh ht hj)
    exact (finish_word c b.tensor _ admit fresh _ (prepareBackward b later) _ hm).trans (source.value h hh t ht j hj)

theorem backward_preserves (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (later : ByteMemory) (checks : backwardChecks c f b=Bool.true)
    (t : Tensor) (fresh : ∀ a∈Separate.backwardFootprint c,Disjoint (b.tensor a.region) t)
    (zeroFresh : b.incomingPresent=Bool.false → Disjoint b.incoming t)
    (point : Fin (2^64)) (hlo : t.storageBase≤point.val) (hhi : point.val<t.storageBase+t.storageBytes) :
    backwardMemory c f b p tree later point=later point := by
  unfold backwardMemory
  rw [finish_preserves_allocation c b.tensor _ (backward_store_admission c f b checks) _ _ t fresh point hlo hhi]
  cases hi:b.incomingPresent
  · simp only [prepareBackward,hi,↓reduceIte]
    exact zero_preserves_allocation _ b.incoming t (backward_pointer c f b checks .incoming (by simp))
      (zeroFresh hi) later point hlo hhi
  · simp only [prepareBackward,hi,↓reduceIte]

/-- Actual later backward call. Saved-history values are derived from preceding
forward bytes and ordinary saved-storage preservation. Native laws occur only
in the numerical reference result; source addresses and raw frames do not use them. -/
theorem backward_native (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (b : BackwardObjects (checkedShape g hs) f)
    (actual reference : Primitives) (tree : SumTree) (complete : tree.Complete (checkedShape g hs).keyBlocks)
    (priorDevice : Nat) (before later : ByteMemory) (fc bc : List Tensor)
    (fchecks : forwardChecks (checkedShape g hs) f=Bool.true)
    (bchecks : backwardChecks (checkedShape g hs) f b=Bool.true)
    (falloc : ForwardAllocation (checkedShape g hs) f fc) (balloc : BackwardAllocation (checkedShape g hs) f b bc)
    (saved : SavedBytes (checkedShape g hs) f (forwardMemory (checkedShape g hs) f actual tree before) later)
    (numeric : LocalLaws actual reference) :
    ∃ result,backwardCall (checkedShape g hs) f b actual tree priorDevice later=some result ∧
      result.dq=b.dq ∧ result.dk=b.dk ∧ result.dv=b.dv ∧ result.seedGradient=none ∧ result.flagGradient=none ∧
      result.launches=backwardLaunches (checkedShape g hs) ∧ result.launchDevice=f.q.device ∧ result.restoredDevice=priorDevice ∧
      BackwardValues (checkedShape g hs) f b reference tree before later result.memory ∧
      (∀ a∈Separate.backwardFootprint (checkedShape g hs),∀ j : Fin 4,
        extendMemory result.memory (byteAddress (checkedShape g hs) b.tensor (a,j))=
          encode (Separate.backward (nativeOps actual) (checkedShape g hs) (nativeScale actual (checkedShape g hs)) tree
            (backwardHeap (checkedShape g hs) f b later) a) j) ∧
      (∀ point : Fin (2^64),
        (∀ i∈byteEvents (Separate.backwardFootprint (checkedShape g hs)),point.val≠byteAddress (checkedShape g hs) b.tensor i) →
        (b.incomingPresent=Bool.false → ¬(b.incoming.dataPointer≤point.val ∧ point.val<b.incoming.dataPointer+4*b.incoming.numel)) →
        result.memory point=later point) ∧
      (∀ t∈bc,∀ point : Fin (2^64),t.storageBase≤point.val → point.val<t.storageBase+t.storageBytes →
        result.memory point=later point) ∧
      (∀ t∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList,∀ point : Fin (2^64),
        t.storageBase≤point.val → point.val<t.storageBase+t.storageBytes → result.memory point=later point) ∧
      (b.incomingPresent=Bool.true → ∀ point : Fin (2^64),b.incoming.storageBase≤point.val →
        point.val<b.incoming.storageBase+b.incoming.storageBytes → result.memory point=later point) ∧
      (∀ t,b.terminal=some t → ∀ point : Fin (2^64),t.storageBase≤point.val →
        point.val<t.storageBase+t.storageBytes → result.memory point=later point) := by
  have admit:=backward_store_admission (checkedShape g hs) f b bchecks
  have sourceSaved:=saved_from_forward (checkedShape g hs) f b actual tree before later fc bc fchecks bchecks falloc balloc saved
  refine ⟨backwardReturn _ f b actual tree priorDevice later,?_,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [backwardCall,bchecks]
  · rw [←local_eq actual reference numeric]
    exact backward_values _ f b actual tree complete before later bchecks balloc.writes sourceSaved
  · intro a ha j
    exact finish_byte _ b.tensor _ admit balloc.writes _ _ a ha j
  · intro point outside zeroOutside
    change finishBytes _ _ _ _ _ point=later point
    rw [finish_frame _ _ _ _ _ point outside]
    cases hi:b.incomingPresent
    · simp only [prepareBackward,hi,↓reduceIte]
      exact zero_frame b.incoming later point (zeroOutside hi)
    · simp only [prepareBackward,hi,↓reduceIte]
  · intro t ht point hlo hhi
    exact backward_preserves _ f b actual tree later bchecks t
      (fun a ha=>balloc.caller a ha t ht) (fun hn=>balloc.zeroCaller hn t ht) point hlo hhi
  · intro t ht point hlo hhi
    exact backward_preserves _ f b actual tree later bchecks t
      (fun a ha=>balloc.saved a ha t ht) (fun hn=>balloc.zeroSaved hn t ht) point hlo hhi
  · intro hi point hlo hhi
    exact backward_preserves _ f b actual tree later bchecks b.incoming (balloc.incoming hi)
      (fun hn=>False.elim (Bool.noConfusion (hi.symm.trans hn))) point hlo hhi
  · intro t ht point hlo hhi
    exact backward_preserves _ f b actual tree later bchecks t (balloc.terminal t ht)
      (fun hn=>balloc.zeroTerminal hn t ht) point hlo hhi

#axiomsClean backward_values
#axiomsClean backward_preserves
#axiomsClean backward_native
#specNonCircular BackwardValues avoiding [backwardCall,backwardMemory,backward_native]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
