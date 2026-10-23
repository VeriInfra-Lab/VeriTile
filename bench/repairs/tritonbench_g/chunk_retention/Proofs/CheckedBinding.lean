import bench.repairs.tritonbench_g.chunk_retention.Proofs.ByteStores
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.SeparatePhysical

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
open HostABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

/-- Actual objects after the original contiguous decorator. The three output
fields are ordinary allocations; no backward object or cotangent is present. -/
structure ForwardObjects (c : Shape) where
  q : Tensor
  k : Tensor
  v : Tensor
  seed : Option Tensor
  history : Tensor
  output : Tensor
  finalState : Option Tensor
  flag : Bool
  finalFlag : finalState.isSome=flag
  historyMeta : denseChecks (shapeHistory c) q.device history=Bool.true
  outputMeta : denseChecks (shapeValue c) q.device output=Bool.true
  finalMeta : ∀ t,finalState=some t → denseChecks (shapeState c) q.device t=Bool.true

def ForwardObjects.tensor (f : ForwardObjects c) : Region→Tensor
  | .q=>f.q | .k=>f.k | .v=>f.v | .seed=>f.seed.getD f.q
  | .history=>f.history | .output=>f.output | .finalState=>f.finalState.getD f.q
  | _=>f.q

def forwardRegions (flag : Bool) : List Region := [.history,.output]++if flag then [.finalState] else []
def forwardPointers (c : Shape) (f : ForwardObjects c) : List (Nat×Tensor) :=
  [(keyExtent c,f.q),(keyExtent c,f.k),(valueExtent c,f.v),(historyExtent c,f.history),(valueExtent c,f.output)]++
    f.seed.toList.map (fun t=>(stateExtent c,t))++f.finalState.toList.map (fun t=>(stateExtent c,t))

def forwardChecks (c : Shape) (f : ForwardObjects c) :=
  planChecks c && inputChecks c f.q f.k f.v f.seed &&
    (forwardPointers c f).all (fun item=>pointerChecks item.1 item.2)

/-- Actual source loads; only absent optional buffers are virtual zero. Ordinary
contiguous storage uses source canonical offsets, including singleton strides. -/
def heapOfBytes (c : Shape) (objects : Region→Tensor) (absent : Region→Bool) (m : ByteMemory) : Heap Word :=
  fun a=>if a.Valid c ∧ absent a.region=Bool.false then readWord m (cellAddress c objects a) else 0

def forwardAbsent (f : ForwardObjects c) : Region→Bool
  | .seed=>f.seed.isNone | .incoming | .terminal=>Bool.true | _=>Bool.false

def forwardHeap (c : Shape) (f : ForwardObjects c) (m : ByteMemory) := heapOfBytes c f.tensor (forwardAbsent f) m

theorem forward_event_region (c : Shape) (flag : Bool) (a : Address)
    (ha : a∈Separate.forwardFootprint c flag) : a.region∈forwardRegions flag := by
  simp only [Separate.forwardFootprint,List.mem_append] at ha
  rcases ha with ha|ha
  · have hr:=Source.h_footprint c flag a ha
    simp only [Phase.writes,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hh|hf
    · simp [forwardRegions,hh]
    · have hflag : flag=Bool.true := by
        cases flag
        · simp only [Source.eventsH,List.mem_flatMap,Source.programs,List.mem_range,List.mem_append,
            Bool.false_eq_true,↓reduceIte,List.not_mem_nil,or_false] at ha
          rcases ha with ⟨h,hh,vb,hv,kb,hk,t,ht,ha⟩
          have bad:=(Source.tile_bounds _ _ _ _ _ _ _ _ _ a ha).1
          rw [hf] at bad
          contradiction
        · rfl
      simp [forwardRegions,hflag,hf]
  · have hr:=Source.o_footprint c a ha
    simp only [Phase.writes,List.mem_singleton] at hr
    simp [forwardRegions,hr]

theorem forward_event_valid (c : Shape) (flag : Bool) (a : Address)
    (ha : a∈Separate.forwardFootprint c flag) : a.Valid c := by
  rcases List.mem_append.mp ha with hh|ho
  · exact Source.h_valid c flag a hh
  · exact Source.o_valid c a ho

theorem forward_output_meta (c : Shape) (f : ForwardObjects c) (r : Region)
    (hr : r∈forwardRegions f.flag) : denseChecks (regionShape c r) f.q.device (f.tensor r)=Bool.true := by
  simp only [forwardRegions,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with (rfl|rfl)|hf
  · exact f.historyMeta
  · exact f.outputMeta
  · have hflag : f.flag=Bool.true := by cases h:f.flag <;> simp [h] at hf ⊢
    have hr : r=Region.finalState := by simpa [hflag] using hf
    subst r
    have hs : f.finalState.isSome=Bool.true := f.finalFlag.trans hflag
    cases he:f.finalState with
    | none=>simp [he] at hs
    | some t=>simpa [ForwardObjects.tensor,he] using f.finalMeta t he

theorem forward_pointer (c : Shape) (f : ForwardObjects c) (checks : forwardChecks c f=Bool.true)
    (r : Region) (hr : r∈forwardRegions f.flag) : pointerChecks (extent c r) (f.tensor r)=Bool.true := by
  simp only [forwardChecks,Bool.and_eq_true,List.all_eq_true] at checks
  apply checks.2 (extent c r,f.tensor r)
  simp only [forwardRegions,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with (rfl|rfl)|hf
  · simp [forwardPointers,ForwardObjects.tensor,extent]
  · simp [forwardPointers,ForwardObjects.tensor,extent]
  · have hflag : f.flag=Bool.true := by cases h:f.flag <;> simp [h] at hf ⊢
    have hr : r=Region.finalState := by simpa [hflag] using hf
    subst r
    have hs : f.finalState.isSome=Bool.true := f.finalFlag.trans hflag
    cases he:f.finalState with
    | none=>simp [he] at hs
    | some t=>simp [forwardPointers,ForwardObjects.tensor,extent,he]

theorem forward_store_admission (c : Shape) (f : ForwardObjects c) (checks : forwardChecks c f=Bool.true) :
    StoreAdmission c f.tensor (Separate.forwardFootprint c f.flag) := by
  refine ⟨forward_event_valid c f.flag,?_,?_⟩
  · intro a ha;exact forward_pointer c f checks a.region (forward_event_region c f.flag a ha)
  · intro a ha
    have h:=forward_output_meta c f a.region (forward_event_region c f.flag a ha)
    simp only [denseChecks,decide_eq_true_eq] at h
    exact h.2.2.2.2.2.2.2.2.2.trans (region_count c a.region)

/-- Actual fresh forward allocations preserve every pre-existing caller
allocation, including raw views that the original decorator copied. -/
structure ForwardAllocation (c : Shape) (f : ForwardObjects c) (caller : List Tensor) : Prop where
  writes : SeparateStores f.tensor (Separate.forwardFootprint c f.flag)
  saved : ∀ a∈Separate.forwardFootprint c f.flag,∀ t∈[f.q,f.k,f.v]++f.seed.toList,
    Disjoint (f.tensor a.region) t
  caller : ∀ a∈Separate.forwardFootprint c f.flag,∀ t∈caller,Disjoint (f.tensor a.region) t

#axiomsClean forward_event_region
#axiomsClean forward_event_valid
#axiomsClean forward_output_meta
#axiomsClean forward_pointer
#axiomsClean forward_store_admission
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
