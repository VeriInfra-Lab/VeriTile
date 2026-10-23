import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardReturn
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Calls

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3500000

/-- Unrestricted Real contents at the actual byte-addressed Tensor locations.
This erasure never factors through finite floating-point word decoding. -/
abbrev RealMemory := Nat→ℝ

def heapOfReal (c : Shape) (objects : Region→Tensor) (absent : Region→Bool) (m : RealMemory) : Heap ℝ :=
  fun a=>if a.Valid c ∧ absent a.region=Bool.false then m (cellAddress c objects a) else 0

def forwardRealHeap (c : Shape) (f : ForwardObjects c) (m : RealMemory) :=
  heapOfReal c f.tensor (forwardAbsent f) m

def zeroReal (t : Tensor) (m : RealMemory) : RealMemory :=
  fun point=>if t.dataPointer≤point ∧ point<t.dataPointer+4*t.numel then 0 else m point

def prepareReal (b : BackwardObjects c f) (m : RealMemory) : RealMemory :=
  if b.incomingPresent then m else zeroReal b.incoming m

def backwardRealHeap (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) (m : RealMemory) :=
  heapOfReal c b.tensor (backwardAbsent b) (prepareReal b m)

def finishReal (c : Shape) (objects : Region→Tensor) (events : List Address)
    (after : Heap ℝ) (before : RealMemory) : RealMemory :=
  storeAll (cellAddress c objects) after before events

theorem finish_real_read (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (fresh : SeparateStores objects events)
    (after : Heap ℝ) (before : RealMemory) (a : Address) (ha : a∈events) :
    finishReal c objects events after before (cellAddress c objects a)=after a := by
  apply store_read _ _ _ _ a ha
  intro b hb eq
  have he : byteAddress c objects (b,0)=byteAddress c objects (a,0) := by simpa [byteAddress] using eq
  have hi:=byte_injective c objects events checks fresh (b,0) (a,0)
    ((byteEvents_mem _ _ _).mpr hb) ((byteEvents_mem _ _ _).mpr ha) he
  exact congrArg after (congrArg Prod.fst hi)

theorem finish_real_frame (c : Shape) (objects : Region→Tensor) (events : List Address)
    (after : Heap ℝ) (before : RealMemory) (point : Nat)
    (outside : ∀ a∈events,point≠cellAddress c objects a) :
    finishReal c objects events after before point=before point := store_frame _ _ _ _ _ outside

theorem finish_real_preserves (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (after : Heap ℝ) (before : RealMemory)
    (t : Tensor) (fresh : ∀ a∈events,Disjoint (objects a.region) t)
    (point : Nat) (hlo : t.storageBase≤point) (hhi : point<t.storageBase+t.storageBytes) :
    finishReal c objects events after before point=before point := by
  apply finish_real_frame
  intro a ha eq
  have bounds:=byte_bounds c objects events checks (a,0) ((byteEvents_mem _ _ _).mpr ha)
  have hf:=fresh a ha
  simp only [byteAddress,Fin.val_zero,Nat.add_zero] at bounds
  unfold Disjoint at hf
  omega

theorem zero_real_preserves (extent : Nat) (z t : Tensor) (checks : pointerChecks extent z=Bool.true)
    (fresh : Disjoint z t) (m : RealMemory) (point : Nat)
    (hlo : t.storageBase≤point) (hhi : point<t.storageBase+t.storageBytes) : zeroReal z m point=m point := by
  unfold zeroReal
  apply if_neg
  simp only [pointerChecks,decide_eq_true_eq] at checks
  have hp:=z.pointer
  rw [checks.1] at hp
  unfold Disjoint at fresh
  omega

def forwardRealMemory (c : Shape) (f : ForwardObjects c) (scale : ℝ) (tree : SumTree) (before : RealMemory) : RealMemory :=
  finishReal c f.tensor (Separate.forwardFootprint c f.flag)
    (Source.afterO realOps c scale tree f.flag (forwardRealHeap c f before)) before

def backwardRealMemory (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (later : RealMemory) : RealMemory :=
  finishReal c b.tensor (Separate.backwardFootprint c)
    (Separate.backward realOps c scale tree (backwardRealHeap c f b later)) (prepareReal b later)

def SavedReal (c : Shape) (f : ForwardObjects c) (snapshot later : RealMemory) : Prop :=
  ∀ r∈savedRegions,∀ point,(f.tensor r).storageBase≤point →
    point<(f.tensor r).storageBase+(f.tensor r).storageBytes → later point=snapshot point

theorem prepared_real_saved (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) (caller : List Tensor)
    (checks : backwardChecks c f b=Bool.true) (allocation : BackwardAllocation c f b caller)
    (snapshot later : RealMemory) (saved : SavedReal c f snapshot later)
    (r : Region) (hr : r∈savedRegions) (point : Nat)
    (hlo : (f.tensor r).storageBase≤point)
    (hhi : point<(f.tensor r).storageBase+(f.tensor r).storageBytes) :
    prepareReal b later point=snapshot point := by
  have ht : f.tensor r∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList := by
    simp only [savedRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl <;> simp [ForwardObjects.tensor]
  have hp:=backward_pointer c f b checks .incoming (by simp)
  cases hi:b.incomingPresent
  · simp only [prepareReal,hi,↓reduceIte]
    exact (zero_real_preserves _ b.incoming (f.tensor r) hp (allocation.zeroSaved hi _ ht) later point hlo hhi).trans
      (saved r hr point hlo hhi)
  · simpa only [prepareReal,hi,↓reduceIte] using saved r hr point hlo hhi

theorem prepared_real_cell (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (caller : List Tensor) (fchecks : forwardChecks c f=Bool.true) (bchecks : backwardChecks c f b=Bool.true)
    (allocation : BackwardAllocation c f b caller) (snapshot later : RealMemory)
    (saved : SavedReal c f snapshot later) (a : Address) (ha : a.Valid c) (hr : a.region∈savedRegions) :
    backwardRealHeap c f b later a=forwardRealHeap c f snapshot a := by
  have hp:=backward_pointer c f b bchecks a.region (by simp only [List.mem_append];exact Or.inl (Or.inr hr))
  rw [backward_saved_tensor c f b a.region hr] at hp
  have bounds:=active_bytes _ _ hp (a.offset c)
    (by rw [saved_count c f fchecks a.region hr];exact Address.offset_bound c a ha) 0 (by decide)
  have hm:=saved_absent c f b a.region hr
  simp only [backwardRealHeap,forwardRealHeap,heapOfReal,ha,hm.1,hm.2,and_self,↓reduceIte,cellAddress,
    backward_saved_tensor c f b a.region hr]
  exact prepared_real_saved c f b caller bchecks allocation snapshot later saved a.region hr _
    (by simpa only [Nat.add_zero] using bounds.1) (by simpa only [Nat.add_zero] using bounds.2.1)

theorem forward_input_real (c : Shape) (f : ForwardObjects c) (scale : ℝ) (tree : SumTree)
    (before : RealMemory) (caller : List Tensor) (checks : forwardChecks c f=Bool.true)
    (allocation : ForwardAllocation c f caller) (a : Address) (ha : a.Valid c)
    (hr : a.region∈[Region.q,.k,.v]) :
    forwardRealHeap c f (forwardRealMemory c f scale tree before) a=forwardRealHeap c f before a := by
  have saved : a.region∈savedRegions := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr <;> simp [savedRegions,hr]
  have hp : pointerChecks (extent c a.region) (f.tensor a.region)=Bool.true := by
    have hc:=checks
    simp only [forwardChecks,Bool.and_eq_true,List.all_eq_true] at hc
    apply hc.2 (extent c a.region,f.tensor a.region)
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr <;> simp [forwardPointers,ForwardObjects.tensor,extent,hr]
  have hm : forwardAbsent f a.region=Bool.false := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr <;> simp [forwardAbsent,hr]
  have bounds:=active_bytes _ _ hp (a.offset c)
    (by rw [saved_count c f checks a.region saved];exact Address.offset_bound c a ha) 0 (by decide)
  simp only [forwardRealHeap,heapOfReal,ha,hm,and_self,↓reduceIte,cellAddress]
  apply finish_real_preserves c f.tensor _ (forward_store_admission c f checks) _ before (f.tensor a.region)
  · intro b hb
    apply allocation.saved b hb (f.tensor a.region)
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr <;> simp [ForwardObjects.tensor,hr]
  · simpa only [Nat.add_zero] using bounds.1
  · simpa only [Nat.add_zero] using bounds.2.1

/-- Saved mathematical history is derived from actual preceding stores and
ordinary saved allocation preservation, just as in the raw-byte endpoint. -/
theorem saved_real_from_forward (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (before later : RealMemory) (fc bc : List Tensor)
    (fchecks : forwardChecks c f=Bool.true) (bchecks : backwardChecks c f b=Bool.true)
    (falloc : ForwardAllocation c f fc) (balloc : BackwardAllocation c f b bc)
    (saved : SavedReal c f (forwardRealMemory c f scale tree before) later) :
    Separate.Saved realOps c scale tree f.flag (forwardRealHeap c f before) (backwardRealHeap c f b later) := by
  have same (r : Region) (hr : r∈[Region.q,.k,.v]) (h t e : Nat) :
      backwardRealHeap c f b later ⟨r,h,0,t,e⟩=forwardRealHeap c f before ⟨r,h,0,t,e⟩ := by
    have hs : r∈savedRegions := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl <;> simp [savedRegions]
    by_cases ha : (⟨r,h,0,t,e⟩:Address).Valid c
    · exact (prepared_real_cell c f b bc fchecks bchecks balloc _ later saved _ ha hs).trans
        (forward_input_real c f scale tree before fc fchecks falloc _ ha hr)
    · simp [backwardRealHeap,forwardRealHeap,heapOfReal,ha]
  refine ⟨?_,?_,?_,?_⟩
  · funext h t e;exact same .q (by simp) h t e
  · funext h t e;exact same .k (by simp) h t e
  · funext h t e;exact same .v (by simp) h t e
  · intro h hh t ht e he j hj
    rw [prepared_real_cell c f b bc fchecks bchecks balloc _ later saved ⟨.history,h,t,e,j⟩ ⟨hh,ht,he,hj⟩ (by simp [savedRegions])]
    simp only [forwardRealHeap,heapOfReal,Address.Valid,hh,ht,he,hj,and_self,forwardAbsent,↓reduceIte]
    exact finish_real_read c f.tensor _ (forward_store_admission c f fchecks) falloc.writes _ before _
      (List.mem_append_left _ (Source.history_cover c f.flag h t e j hh ht he hj))

#axiomsClean finish_real_read
#axiomsClean finish_real_preserves
#axiomsClean prepared_real_cell
#axiomsClean forward_input_real
#axiomsClean saved_real_from_forward
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
