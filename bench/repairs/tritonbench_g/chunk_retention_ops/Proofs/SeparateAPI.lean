import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Separate
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Separate

def backwardAPI (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (ri : Bool) (later : Heap α) :=
  Source.execute ops c scale tree .sum (Source.eventsSum c)
    (Source.execute ops c scale tree .qkv (Source.eventsQKV c)
      (Source.execute ops c scale tree .dh
        (Source.eventsDH c ++ if ri then Source.eventsInitial c else []) later))

theorem backwardAPI_agrees (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (ri : Bool) (later : Heap α) : Source.ExceptInitial
    (backwardAPI ops c scale tree ri later) (backward ops c scale tree later) := by
  apply Source.execute_except_initial
  apply Source.execute_except_initial
  intro point hp
  rw [Source.execute_append]
  cases ri
  · rfl
  · rw [Source.execute_spec ops c scale tree .dh _ (fun a ha => by
      rw [Source.initial_footprint c a ha]; simp [Phase.writes])]
    simp only [↓reduceIte]
    rw [if_neg (fun ha => hp (Source.initial_footprint c point ha))]
    rfl

theorem backwardAPI_initial (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (later : Heap α) (h e j : Nat) (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    backwardAPI ops c scale tree Bool.true later ⟨.initialGradient,h,0,e,j⟩=
      ops.cast (Reference.reverseCarry ops c (inputsOfHeap later) scale h c.chunks e j) := by
  rw [backwardAPI,Source.execute_frame ops c scale tree .sum _ (Source.sum_footprint c) _ _ (by simp [Phase.writes]),
    Source.execute_frame ops c scale tree .qkv _ (Source.qkv_footprint c) _ _ (by simp [Phase.writes]),
    Source.execute_spec ops c scale tree .dh _ (Source.initial_events_writes c Bool.true)]
  have hm : (⟨.initialGradient,h,0,e,j⟩:Address)∈Source.eventsDH c ++ if Bool.true then Source.eventsInitial c else [] := by
    apply List.mem_append_right
    exact Source.initial_cover c h e j hh he hj
  rw [if_pos hm]
  simp only [Source.rhs,Source.bodyRHS,Source.project_inputs ops .dh (by decide),reduceCtorEq,↓reduceIte]

theorem backwardAPI_post (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (rf ri : Bool)
    (before later : Heap α) (saved : Saved ops c scale tree rf before later) :
    BackwardPost ops c (joined before later) scale tree (backwardAPI ops c scale tree ri later) := by
  have h:=backward_post ops c scale tree complete rf before later saved
  have same:=backwardAPI_agrees ops c scale tree ri later
  exact ⟨fun a ha t ht e he => (same _ (by simp)).trans (h.query a ha t ht e he),
    fun a ha t ht e he => (same _ (by simp)).trans (h.key a ha t ht e he),
    fun a ha t ht j hj => (same _ (by simp)).trans (h.value a ha t ht j hj)⟩

def forwardPhysical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf : Bool) (entry : Memory α) :=
  Source.executePhysical ops c b scale tree .o (Source.eventsO c)
    (Source.executePhysical ops c b scale tree .h (Source.eventsH c rf) entry)
def backwardPhysical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (ri : Bool) (later : Memory α) :=
  Source.executePhysical ops c b scale tree .sum (Source.eventsSum c)
    (Source.executePhysical ops c b scale tree .qkv (Source.eventsQKV c)
      (Source.executePhysical ops c b scale tree .dh
        (Source.eventsDH c ++ if ri then Source.eventsInitial c else []) later))

theorem forward_physical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf : Bool) (entry : Memory α) :
    b.view ops.zero (forwardPhysical ops c b scale tree rf entry)=
      Source.afterO ops c scale tree rf (b.view ops.zero entry) := by
  unfold forwardPhysical Source.afterO Source.afterH
  rw [Source.execute_physical ops c b scale tree .o _ (Source.o_valid c)
        (fun a ha=>Source.write_not_input .o a (Source.o_footprint c a ha)),
      Source.execute_physical ops c b scale tree .h _ (Source.h_valid c rf)
        (fun a ha=>Source.write_not_input .h a (Source.h_footprint c rf a ha))]

theorem backward_physical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (ri : Bool) (later : Memory α) :
    b.view ops.zero (backwardPhysical ops c b scale tree ri later)=
      backwardAPI ops c scale tree ri (b.view ops.zero later) := by
  unfold backwardPhysical backwardAPI
  rw [Source.execute_physical ops c b scale tree .sum _ (Source.sum_valid c)
        (fun a ha=>Source.write_not_input .sum a (Source.sum_footprint c a ha)),
      Source.execute_physical ops c b scale tree .qkv _ (Source.qkv_valid c)
        (fun a ha=>Source.write_not_input .qkv a (Source.qkv_footprint c a ha)),
      Source.execute_physical ops c b scale tree .dh _ (Source.dhAPI_valid c ri)
        (fun a ha=>Source.write_not_input .dh a (Source.initial_events_writes c ri a ha))]

theorem forward_physical_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf : Bool) (entry : Memory α) (point : Nat)
    (outside : ∀ a∈forwardFootprint c rf,point≠b.address a) :
    forwardPhysical ops c b scale tree rf entry point=entry point := by
  unfold forwardPhysical
  rw [Source.physical_frame ops c b scale tree .o _ _ point (fun a ha=>outside a (by simp [forwardFootprint,ha])),
    Source.physical_frame ops c b scale tree .h _ _ point (fun a ha=>outside a (by simp [forwardFootprint,ha]))]

theorem backward_physical_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (ri : Bool) (later : Memory α) (point : Nat)
    (outside : ∀ a∈backwardFootprint c ++ if ri then Source.eventsInitial c else [],point≠b.address a) :
    backwardPhysical ops c b scale tree ri later point=later point := by
  unfold backwardPhysical
  rw [Source.physical_frame ops c b scale tree .sum _ _ point (fun a ha=>outside a (by simp [backwardFootprint,ha])),
    Source.physical_frame ops c b scale tree .qkv _ _ point (fun a ha=>outside a (by simp [backwardFootprint,ha])),
    Source.physical_frame ops c b scale tree .dh _ _ point (fun a ha=>outside a (by
      rcases List.mem_append.mp ha with ha|ha
      · simp [backwardFootprint,ha]
      · exact List.mem_append_right _ ha))]

/-- Actual Tensor handles are represented by their data-pointer word base and
metadata. Runtime binds these descriptors; no computed value is a premise. -/
structure Handle where
  base : Nat
  metadata : HostABI.TensorMeta

def outputHandle (c : Shape) (b : Binding c) (device : Nat) : Handle :=
  ⟨b.base .output,HostABI.valueMeta c device 0⟩
def stateHandle (c : Shape) (b : Binding c) (device : Nat) : Handle :=
  ⟨b.base .finalState,HostABI.stateMeta c device⟩
def gradientHandles (c : Shape) (b : Binding c) (device : Nat) (ri : Bool) : List (Option Handle) :=
  [some ⟨b.base .dq,HostABI.keyMeta c device 0⟩,some ⟨b.base .dk,HostABI.keyMeta c device 0⟩,
   some ⟨b.base .dv,HostABI.valueMeta c device 0⟩,
   if ri then some ⟨b.base .initialGradient,HostABI.stateMeta c device⟩ else none,
   none,none,none]

/-- Saved Tensor/version binding is checked at backward entry. It connects the
actual later physical Q/K/V/history reads to forward data; the later cotangent
values are unrestricted. Checkpoint recomputation may establish the same
history condition. Forward output and unrelated later memory may change. -/
structure SavedContext (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (rf : Bool)
    (forwardBinding backwardBinding : Binding c) (before later : Memory α) : Prop where
  handles : ∀ r,r∈[Region.q,Region.k,Region.v] → forwardBinding.base r=backwardBinding.base r
  versions : Saved ops c scale tree rf
    (forwardBinding.view ops.zero before) (backwardBinding.view ops.zero later)

#axiomsClean backwardAPI_post
#axiomsClean backwardAPI_initial
#axiomsClean forward_physical
#axiomsClean backward_physical
#axiomsClean forward_physical_frame
#axiomsClean backward_physical_frame
end Separate
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

