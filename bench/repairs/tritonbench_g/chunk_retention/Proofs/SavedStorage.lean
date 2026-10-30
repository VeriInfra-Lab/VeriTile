import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardBinding

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3500000

def SavedBytes (c : Shape) (f : ForwardObjects c) (snapshot later : ByteMemory) : Prop :=
  ∀ r∈savedRegions,∀ point : Fin (2^64),(f.tensor r).storageBase≤point.val →
    point.val<(f.tensor r).storageBase+(f.tensor r).storageBytes → later point=snapshot point

theorem saved_meta (c : Shape) (f : ForwardObjects c) (checks : forwardChecks c f=Bool.true)
    (r : Region) (hr : r∈savedRegions) : denseChecks (regionShape c r) f.q.device (f.tensor r)=Bool.true := by
  simp only [forwardChecks,Bool.and_eq_true] at checks
  have hc:=checks.1.2
  simp only [inputChecks,Bool.and_eq_true] at hc
  simp only [savedRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl
  · exact hc.1.1.1
  · exact hc.1.1.2
  · exact hc.1.2
  · exact f.historyMeta

theorem saved_count (c : Shape) (f : ForwardObjects c) (checks : forwardChecks c f=Bool.true)
    (r : Region) (hr : r∈savedRegions) : (f.tensor r).numel=r.capacity c := by
  have h:=saved_meta c f checks r hr
  simp only [denseChecks,decide_eq_true_eq] at h
  exact h.2.2.2.2.2.2.2.2.2.trans (region_count c r)

theorem backward_saved_tensor (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (r : Region) (hr : r∈savedRegions) : b.tensor r=f.tensor r := by
  simp only [savedRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl <;> rfl

theorem saved_absent (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (r : Region) (hr : r∈savedRegions) : forwardAbsent f r=Bool.false ∧ backwardAbsent b r=Bool.false := by
  simp only [savedRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl <;> exact ⟨rfl,rfl⟩

theorem prepared_saved (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) (caller : List Tensor)
    (checks : backwardChecks c f b=Bool.true) (allocation : BackwardAllocation c f b caller)
    (snapshot later : ByteMemory) (saved : SavedBytes c f snapshot later)
    (r : Region) (hr : r∈savedRegions) (point : Fin (2^64))
    (hlo : (f.tensor r).storageBase≤point.val)
    (hhi : point.val<(f.tensor r).storageBase+(f.tensor r).storageBytes) :
    prepareBackward b later point=snapshot point := by
  have ht : f.tensor r∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList := by
    simp only [savedRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl <;> simp [ForwardObjects.tensor]
  have hp:=backward_pointer c f b checks .incoming (by simp)
  cases hi:b.incomingPresent
  · simp only [prepareBackward,hi,↓reduceIte]
    change zeroTensor b.incoming later point=snapshot point
    exact (zero_preserves_allocation _ b.incoming (f.tensor r) hp (allocation.zeroSaved hi _ ht) later point hlo hhi).trans
      (saved r hr point hlo hhi)
  · simpa only [prepareBackward,hi,↓reduceIte] using saved r hr point hlo hhi

theorem prepared_saved_word (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (caller : List Tensor) (fchecks : forwardChecks c f=Bool.true) (bchecks : backwardChecks c f b=Bool.true)
    (allocation : BackwardAllocation c f b caller) (snapshot later : ByteMemory)
    (saved : SavedBytes c f snapshot later) (a : Address) (ha : a.Valid c) (hr : a.region∈savedRegions) :
    backwardHeap c f b later a=forwardHeap c f snapshot a := by
  have hp:=backward_pointer c f b bchecks a.region (by simp only [List.mem_append];exact Or.inl (Or.inr hr))
  rw [backward_saved_tensor c f b a.region hr] at hp
  have hm:=saved_absent c f b a.region hr
  simp only [backwardHeap,forwardHeap,heapOfBytes,ha,hm.1,hm.2,and_self,↓reduceIte,cellAddress,
    backward_saved_tensor c f b a.region hr]
  exact read_preserved _ (f.tensor a.region) hp snapshot (prepareBackward b later)
    (prepared_saved c f b caller bchecks allocation snapshot later saved a.region hr) (a.offset c)
    (by rw [saved_count c f fchecks a.region hr];exact Address.offset_bound c a ha)

theorem forward_input_word (c : Shape) (f : ForwardObjects c) (p : Primitives) (tree : SumTree)
    (before : ByteMemory) (caller : List Tensor) (checks : forwardChecks c f=Bool.true)
    (allocation : ForwardAllocation c f caller) (a : Address) (ha : a.Valid c)
    (hr : a.region∈[Region.q,.k,.v]) :
    forwardHeap c f (forwardMemory c f p tree before) a=forwardHeap c f before a := by
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
  simp only [forwardHeap,heapOfBytes,ha,hm,and_self,↓reduceIte,cellAddress]
  apply read_preserved _ _ hp before
  · intro point hlo hhi
    apply finish_preserves_allocation c f.tensor _ (forward_store_admission c f checks) _ before _ ?_ point hlo hhi
    intro b hb
    apply allocation.saved b hb (f.tensor a.region)
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr <;> simp [ForwardObjects.tensor,hr]
  · rw [saved_count c f checks a.region saved]
    exact Address.offset_bound c a ha

/-- The numerical saved-history relation is derived from the forward stores
and ordinary byte preservation between calls. It is not a caller-supplied
kernel-output or safety assertion. -/
theorem saved_from_forward (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (p : Primitives) (tree : SumTree) (before later : ByteMemory) (fc bc : List Tensor)
    (fchecks : forwardChecks c f=Bool.true) (bchecks : backwardChecks c f b=Bool.true)
    (falloc : ForwardAllocation c f fc) (balloc : BackwardAllocation c f b bc)
    (saved : SavedBytes c f (forwardMemory c f p tree before) later) :
    Separate.Saved (nativeOps p) c (nativeScale p c) tree f.flag
      (forwardHeap c f before) (backwardHeap c f b later) := by
  have same (r : Region) (hr : r∈[Region.q,.k,.v]) (h t e : Nat) :
      backwardHeap c f b later ⟨r,h,0,t,e⟩=forwardHeap c f before ⟨r,h,0,t,e⟩ := by
    have hs : r∈savedRegions := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl <;> simp [savedRegions]
    by_cases ha : (⟨r,h,0,t,e⟩:Address).Valid c
    · exact (prepared_saved_word c f b bc fchecks bchecks balloc _ later saved _ ha hs).trans
        (forward_input_word c f p tree before fc fchecks falloc _ ha hr)
    · simp [backwardHeap,forwardHeap,heapOfBytes,ha]
  refine ⟨?_,?_,?_,?_⟩
  · funext h t e;exact same .q (by simp) h t e
  · funext h t e;exact same .k (by simp) h t e
  · funext h t e;exact same .v (by simp) h t e
  · intro h hh t ht e he j hj
    rw [prepared_saved_word c f b bc fchecks bchecks balloc _ later saved ⟨.history,h,t,e,j⟩ ⟨hh,ht,he,hj⟩ (by simp [savedRegions])]
    simp only [forwardHeap,heapOfBytes,Address.Valid,hh,ht,he,hj,and_self,forwardAbsent,↓reduceIte]
    exact finish_word c f.tensor _ (forward_store_admission c f fchecks) falloc.writes _ before _
      (List.mem_append_left _ (Source.history_cover c f.flag h t e j hh ht he hj))

#axiomsClean saved_meta
#axiomsClean saved_count
#axiomsClean prepared_saved
#axiomsClean prepared_saved_word
#axiomsClean forward_input_word
#axiomsClean saved_from_forward
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
