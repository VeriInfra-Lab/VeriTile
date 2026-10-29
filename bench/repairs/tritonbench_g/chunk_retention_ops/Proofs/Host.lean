import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Returned
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

def shape (a : Contract.Forward) (h : Contract.ForwardContract a) := Contract.toShape a.dims h.1
def callerInputs (a : Contract.Forward) := [a.q,a.k,a.v]++a.initial.toList

/-- Bind actual observed argument objects and the successful fresh allocations.
No allocation or pointer for an absent/future result is required. -/
structure ForwardCall (a : Contract.Forward) (h : Contract.ForwardContract a) where
  storage : Storage (shape a h)
  device : storage.device=a.k.metadata.device
  q : storage.tensor .q=a.q
  k : storage.tensor .k=a.k
  v : storage.tensor .v=a.v
  primals : ∀ r∈[Region.q,Region.k,Region.v],storage.present r=Bool.true
  initial : ∀ t∈a.initial.toList,storage.tensor .seed=t ∧ storage.present .seed=Bool.true
  seed : storage.zeroInput .seed= !a.initial.isSome
  incoming : storage.zeroInput .incoming=Bool.true
  terminal : storage.zeroInput .terminal=Bool.true
  history : storage.present .history=Bool.true
  output : storage.present .output=Bool.true
  final : a.final=Bool.true → storage.present .finalState=Bool.true

structure BackwardAllocation (c : Shape) (ri checkpoint : Bool) where
  storage : Storage c
  history : storage.present .history=Bool.true
  outputs : ∀ r∈[Region.dhistory,Region.dq,Region.dk,Region.dvPart,Region.dv],storage.present r=Bool.true
  initial : ri=Bool.true → storage.present .initialGradient=Bool.true

theorem fresh_read (storage : Storage c) (r : Region) (read : r∈inputRegions)
    (present : storage.present r=Bool.true) : FreshCaller storage (storage.tensor r) := by
  intro output po write
  apply storage.fresh output r
  · intro eq
    subst output
    exact write read
  · exact po
  · exact present
  · exact Or.inl write

theorem ForwardCall.fresh_inputs (call : ForwardCall a h) (t : Contract.Tensor)
    (member : t∈callerInputs a) : FreshCaller call.storage t := by
  simp only [callerInputs,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (rfl|rfl|rfl)|seed
  · rw [←call.q]
    exact fresh_read _ .q (by simp [inputRegions]) (call.primals _ (by simp))
  · rw [←call.k]
    exact fresh_read _ .k (by simp [inputRegions]) (call.primals _ (by simp))
  · rw [←call.v]
    exact fresh_read _ .v (by simp [inputRegions]) (call.primals _ (by simp))
  · obtain ⟨eq,present⟩ := call.initial t seed
    rw [←eq]
    exact fresh_read _ .seed (by simp [inputRegions]) present

theorem h_optional_tag (c : Shape) (flag : Bool) (a : Address) (mem : a∈Source.eventsH c flag) :
    a.region=.history ∨ (flag=Bool.true ∧ a.region=.finalState) := by
  simp only [Source.eventsH,Source.programs,List.mem_flatMap,List.mem_range] at mem
  rcases mem with ⟨head,hh,vb,hv,kb,hk,mem⟩
  rcases List.mem_append.mp mem with mem|mem
  · rcases List.mem_flatMap.mp mem with ⟨t,ht,mem⟩
    exact Or.inl (Source.tile_tag _ _ _ _ _ _ _ _ _ _ mem)
  · cases flag
    · simp at mem
    · exact Or.inr ⟨rfl,Source.tile_tag _ _ _ _ _ _ _ _ _ _ mem⟩

theorem dh_tag (c : Shape) (a : Address) (mem : a∈Source.eventsDH c) : a.region=.dhistory := by
  simp only [Source.eventsDH,Source.programs,List.mem_flatMap,List.mem_range] at mem
  rcases mem with ⟨head,hh,vb,hv,kb,hk,n,hn,mem⟩
  exact Source.tile_tag _ _ _ _ _ _ _ _ _ _ mem

theorem ForwardCall.allocated (call : ForwardCall a h) (tree : SumTree) (s : Store)
    (mem : s∈forwardCode (shape a h) tree a.final) : call.storage.present s.address.region=Bool.true := by
  have address : s.address∈Separate.forwardFootprint (shape a h) a.final := by
    rw [←forwardCode_addresses (shape a h) tree a.final]
    exact List.mem_map.mpr ⟨s,mem,rfl⟩
  rcases List.mem_append.mp address with hm|om
  · rcases h_optional_tag _ _ _ hm with tag|⟨flag,tag⟩
    · rw [tag]; exact call.history
    · rw [tag]; exact call.final flag
  · have tag := Source.o_footprint _ _ om
    simp only [Phase.writes,List.mem_singleton] at tag
    rw [tag]
    exact call.output

theorem BackwardAllocation.allocated (alloc : BackwardAllocation c ri checkpoint) (tree : SumTree) (s : Store)
    (mem : s∈recomputeCode c tree checkpoint++backwardCode c tree ri) :
    alloc.storage.present s.address.region=Bool.true := by
  have phase : ∀ p events,s∈phaseCode c tree p events → s.address∈events := by
    intro p events hm
    rw [←phaseCode_addresses c tree p events]
    exact List.mem_map.mpr ⟨s,hm,rfl⟩
  rcases List.mem_append.mp mem with recompute|backward
  · cases checkpoint
    · simp [recomputeCode] at recompute
    · have hm := phase .h (Source.eventsH c Bool.false) recompute
      have tag := h_optional_tag c Bool.false s.address hm
      simp only [Bool.false_eq_true,false_and,or_false] at tag
      rw [tag]
      exact alloc.history
  · simp only [backwardCode,List.mem_append] at backward
    rcases backward with (dh|qkv)|sum
    · have mem := phase _ _ dh
      rcases List.mem_append.mp mem with hm|hi
      · rw [dh_tag _ _ hm]
        exact alloc.outputs _ (by simp)
      · cases ri
        · simp at hi
        · have exactTag := Source.initial_footprint c s.address (by simpa using hi)
          rw [exactTag]
          exact alloc.initial rfl
    · have tag := Source.qkv_footprint c s.address (phase _ _ qkv)
      apply alloc.outputs
      simp only [Phase.writes,List.mem_cons,List.not_mem_nil,or_false] at tag ⊢
      tauto
    · have tag := Source.sum_footprint c s.address (phase _ _ sum)
      simp only [Phase.writes,List.mem_singleton] at tag
      rw [tag]
      exact alloc.outputs _ (by simp)

theorem returned_metadata (storage : Storage c) (r : Region) (present : storage.present r=Bool.true) :
    (storage.tensor r).metadata.shape=regionShape c r ∧
    (storage.tensor r).metadata.dtype=.floating .fp32 ∧
    (storage.tensor r).metadata.device=storage.device ∧
    (storage.tensor r).metadata.isContiguous=Bool.true := by
  have h := storage.checked r present
  exact ⟨h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.2.2.2.2.1 rfl⟩

theorem ForwardCall.final_metadata (call : ForwardCall a h) (enabled : a.final=Bool.true) :
    (forwardTensors call.storage a.final).2=some (call.storage.tensor .finalState) ∧
    (call.storage.tensor .finalState).metadata.shape=Contract.stateShape a.dims ∧
    (call.storage.tensor .finalState).metadata.dtype=.floating .fp32 ∧
    (call.storage.tensor .finalState).metadata.device=a.k.metadata.device ∧
    (call.storage.tensor .finalState).metadata.isContiguous=Bool.true := by
  obtain ⟨hs,hd,hdev,hc⟩ := returned_metadata call.storage .finalState (call.final enabled)
  exact ⟨by simp [forwardTensors,enabled],hs,hd,hdev.trans call.device,hc⟩

theorem forward_backing_frame (abi : MemoryABI σ) (ports : Numeric.ScalarPorts)
    (a : Contract.Forward) (h : Contract.ForwardContract a) (call : ForwardCall a h) (scale : Word)
    (compiler : Compiler (shape a h) tree a.scale.value a.initial.isSome a.final terminal ri)
    (before prepared : σ)
    (allocationFrame : ∀ t∈callerInputs a,∀ byte,byte<t.bytes →
      abi.bytes prepared (t.base+byte)=abi.bytes before (t.base+byte)) :
    ∀ t∈callerInputs a,∀ byte,byte<t.bytes →
      abi.bytes (forwardEntry abi ports _ call.storage scale compiler prepared).2 (t.base+byte)=
        abi.bytes before (t.base+byte) := by
  intro t member byte bound
  change abi.bytes (nativeExecute abi ports _ call.storage.binding scale compiler.forward prepared) _=_
  rw [compiler.forward_eq,
    complete_backing_frame abi ports _ call.storage scale _ (forward_stores_valid _ tree a.final)
      (call.allocated tree) prepared t (call.fresh_inputs t member) byte bound,
    allocationFrame t member byte bound]

#axiomsClean ForwardCall.allocated
#axiomsClean BackwardAllocation.allocated
#axiomsClean forward_backing_frame
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
