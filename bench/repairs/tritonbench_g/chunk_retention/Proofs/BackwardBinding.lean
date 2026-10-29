import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardReturn

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
open HostABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

structure BackwardObjects (c : Shape) (f : ForwardObjects c) where
  incoming : Tensor
  incomingPresent : Bool
  terminal : Option Tensor
  dhistory : Tensor
  dq : Tensor
  dk : Tensor
  partialDV : Tensor
  dv : Tensor
  dhistoryMeta : denseChecks (shapeHistory c) f.q.device dhistory=Bool.true
  dqMeta : denseChecks (shapeKey c) f.q.device dq=Bool.true
  dkMeta : denseChecks (shapeKey c) f.q.device dk=Bool.true
  partialMeta : denseChecks [c.keyBlocks,c.B,c.H,c.T,c.V] f.q.device partialDV=Bool.true
  dvMeta : denseChecks (shapeValue c) f.q.device dv=Bool.true

def BackwardObjects.tensor (b : BackwardObjects c f) : Region→Tensor
  | .incoming=>b.incoming | .terminal=>b.terminal.getD f.q
  | .dhistory=>b.dhistory | .dq=>b.dq | .dk=>b.dk | .dvPart=>b.partialDV | .dv=>b.dv
  | r=>f.tensor r

def backwardRegions : List Region := [.dhistory,.dq,.dk,.dvPart,.dv]
def savedRegions : List Region := [.q,.k,.v,.history]

def backwardPointers (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) : List (Nat×Tensor) :=
  [(keyExtent c,f.q),(keyExtent c,f.k),(valueExtent c,f.v),(historyExtent c,f.history),
    (valueExtent c,b.incoming),(historyExtent c,b.dhistory),(keyExtent c,b.dq),(keyExtent c,b.dk),
    (partialExtent c,b.partialDV),(valueExtent c,b.dv)]++b.terminal.toList.map (fun t=>(stateExtent c,t))

def backwardChecks (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) :=
  planChecks c && denseChecks (shapeValue c) f.q.device b.incoming &&
    (match b.terminal with | none=>Bool.true | some t=>f.flag && denseChecks (shapeState c) f.q.device t) &&
    (backwardPointers c f b).all (fun item=>pointerChecks item.1 item.2)

def backwardAbsent (b : BackwardObjects c f) : Region→Bool
  | .seed=>Bool.true | .terminal=>b.terminal.isNone | _=>Bool.false

def prepareBackward (b : BackwardObjects c f) (m : ByteMemory) : ByteMemory :=
  if b.incomingPresent then m else zeroTensor b.incoming m

def backwardHeap (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) (m : ByteMemory) :=
  heapOfBytes c b.tensor (backwardAbsent b) (prepareBackward b m)

theorem backward_event_valid (c : Shape) (a : Address)
    (ha : a∈Separate.backwardFootprint c) : a.Valid c := by
  simp only [Separate.backwardFootprint,List.mem_append] at ha
  rcases ha with (hdh|hqkv)|hs
  · exact Source.dh_valid c a hdh
  · exact Source.qkv_valid c a hqkv
  · exact Source.sum_valid c a hs

theorem backward_event_region (c : Shape) (a : Address)
    (ha : a∈Separate.backwardFootprint c) : a.region∈backwardRegions := by
  simp only [Separate.backwardFootprint,List.mem_append] at ha
  rcases ha with (hdh|hqkv)|hs
  · have hr:=Source.dh_footprint c a hdh
    simp only [Phase.writes,List.mem_singleton] at hr
    simp [backwardRegions,hr]
  · have hr:=Source.qkv_footprint c a hqkv
    simp only [Phase.writes,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr <;> simp [backwardRegions,hr]
  · have hr:=Source.sum_footprint c a hs
    simp only [Phase.writes,List.mem_singleton] at hr
    simp [backwardRegions,hr]

theorem backward_output_meta (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (r : Region) (hr : r∈backwardRegions) : denseChecks (regionShape c r) f.q.device (b.tensor r)=Bool.true := by
  simp only [backwardRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl
  · exact b.dhistoryMeta
  · exact b.dqMeta
  · exact b.dkMeta
  · exact b.partialMeta
  · exact b.dvMeta

theorem backward_pointer (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (checks : backwardChecks c f b=Bool.true) (r : Region) (hr : r∈backwardRegions++savedRegions++[.incoming]) :
    pointerChecks (extent c r) (b.tensor r)=Bool.true := by
  simp only [backwardChecks,Bool.and_eq_true,List.all_eq_true] at checks
  apply checks.2 (extent c r,b.tensor r)
  simp only [backwardRegions,savedRegions,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with ((rfl|rfl|rfl|rfl|rfl)|(rfl|rfl|rfl|rfl))|rfl <;>
    simp [backwardPointers,BackwardObjects.tensor,ForwardObjects.tensor,extent]

theorem backward_store_admission (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (checks : backwardChecks c f b=Bool.true) : StoreAdmission c b.tensor (Separate.backwardFootprint c) := by
  refine ⟨backward_event_valid c,?_,?_⟩
  · intro a ha
    apply backward_pointer c f b checks a.region
    simp only [List.mem_append];exact Or.inl (Or.inl (backward_event_region c a ha))
  · intro a ha
    have h:=backward_output_meta c f b a.region (backward_event_region c a ha)
    simp only [denseChecks,decide_eq_true_eq] at h
    exact h.2.2.2.2.2.2.2.2.2.trans (region_count c a.region)

/-- Both None cotangent handling and ordinary backward allocation preserve the
pre-existing caller/saved allocations. Original incoming aliases are allowed. -/
structure BackwardAllocation (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f) (callers : List Tensor) : Prop where
  writes : SeparateStores b.tensor (Separate.backwardFootprint c)
  caller : ∀ a∈Separate.backwardFootprint c,∀ t∈callers,Disjoint (b.tensor a.region) t
  saved : ∀ a∈Separate.backwardFootprint c,∀ t∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList,
    Disjoint (b.tensor a.region) t
  incoming : b.incomingPresent=Bool.true → ∀ a∈Separate.backwardFootprint c,Disjoint (b.tensor a.region) b.incoming
  terminal : ∀ t,b.terminal=some t → ∀ a∈Separate.backwardFootprint c,Disjoint (b.tensor a.region) t
  zeroCaller : b.incomingPresent=Bool.false → ∀ t∈callers,Disjoint b.incoming t
  zeroSaved : b.incomingPresent=Bool.false → ∀ t∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList,
    Disjoint b.incoming t
  zeroTerminal : b.incomingPresent=Bool.false → ∀ t,b.terminal=some t → Disjoint b.incoming t

#axiomsClean backward_event_valid
#axiomsClean backward_event_region
#axiomsClean backward_output_meta
#axiomsClean backward_pointer
#axiomsClean backward_store_admission
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
