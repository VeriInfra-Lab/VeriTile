import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.HostABI
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section

/-- Valid logical cells for the canonical dense allocations. -/
def Address.Valid (c : Shape) (a : Address) : Prop :=
  a.head<c.heads ∧ match a.region with
  | .q | .k | .dq | .dk => a.slice=0 ∧ a.row<c.T ∧ a.col<c.K
  | .v | .incoming | .output | .dv => a.slice=0 ∧ a.row<c.T ∧ a.col<c.V
  | .seed | .terminal | .finalState | .initialGradient => a.slice=0 ∧ a.row<c.K ∧ a.col<c.V
  | .history | .dhistory => a.slice<c.chunks ∧ a.row<c.K ∧ a.col<c.V
  | .dvPart => a.slice<c.keyBlocks ∧ a.row<c.T ∧ a.col<c.V

def Address.offset (c : Shape) (a : Address) : Nat := match a.region with
  | .q | .k | .dq | .dk => c.keyOffset a.head a.row a.col
  | .v | .incoming | .output | .dv => c.valueOffset a.head a.row a.col
  | .seed | .terminal | .finalState | .initialGradient => c.stateOffset a.head a.row a.col
  | .history | .dhistory => c.historyOffset a.head a.slice a.row a.col
  | .dvPart => c.partialOffset a.slice a.head a.row a.col

def Region.capacity (c : Shape) : Region → Nat
  | .q | .k | .dq | .dk => c.keySize
  | .v | .incoming | .output | .dv => c.valueSize
  | .seed | .terminal | .finalState | .initialGradient => c.stateSize
  | .history | .dhistory => c.historySize
  | .dvPart => c.partialSize

theorem Address.offset_bound (c : Shape) (a : Address) (ha : a.Valid c) :
    a.offset c<a.region.capacity c := by
  rcases a with ⟨r,h,s,i,j⟩
  cases r <;> simp only [Address.Valid] at ha
  all_goals first
  | exact c.key_offset_bound h i j ha.1 ha.2.2.1 ha.2.2.2
  | exact c.value_offset_bound h i j ha.1 ha.2.2.1 ha.2.2.2
  | exact c.state_offset_bound h i j ha.1 ha.2.2.1 ha.2.2.2
  | exact c.history_offset_bound h s i j ha.1 ha.2.1 ha.2.2.1 ha.2.2.2
  | exact c.partial_offset_bound s h i j ha.2.1 ha.1 ha.2.2.1 ha.2.2.2

/-- Physical allocation/alias binding is a structural ABI obligation. It binds
only valid cells, allows arbitrary word-base placement, and is independent of
all numerical operations. Distinct output/input cells must be disjoint. -/
structure Binding (c : Shape) where
  base : Region → Nat
  /-- An absent optional argument has no physical allocation or load. -/
  zeroInput : Region → Bool
  zeroOnlyOptional : ∀ r,zeroInput r=Bool.true → r∈[Region.seed,Region.incoming,Region.terminal]
  injective : ∀ (a b : Address), a.Valid c → b.Valid c →
    (a.region∉inputRegions ∨ b.region∉inputRegions) →
    base a.region+a.offset c=base b.region+b.offset c → a=b

def Binding.address {c : Shape} (b : Binding c) (a : Address) := b.base a.region+a.offset c

def Binding.view {c : Shape} (b : Binding c) (zero : α) (m : Memory α) : Heap α := by
  classical
  exact fun a => if a.Valid c ∧ b.zeroInput a.region=Bool.false then m (b.address a) else zero

def physicalWrite (m : Memory α) (point : Nat) (value : α) : Memory α :=
  fun p => if p=point then value else m p

theorem Binding.writable_live {c : Shape} (b : Binding c) (r : Region)
    (writeOnly : r∉inputRegions) : b.zeroInput r=Bool.false := by
  cases hr : b.zeroInput r
  · rfl
  · have h:=b.zeroOnlyOptional r hr
    have hi : r∈inputRegions := by simp only [List.mem_cons,List.not_mem_nil,or_false] at h; rcases h with h|h|h <;> subst r <;> simp [inputRegions]
    exact False.elim (writeOnly hi)

theorem Binding.view_read {c : Shape} (b : Binding c) (zero : α) (m : Memory α)
    (a : Address) (ha : a.Valid c) (writeOnly : a.region∉inputRegions) :
    b.view zero m a=m (b.address a) := by
  simp [Binding.view,ha,b.writable_live a.region writeOnly]

theorem Binding.view_write {c : Shape} (b : Binding c) (zero : α) (m : Memory α)
    (a : Address) (ha : a.Valid c) (writeOnly : a.region∉inputRegions) (value : α) :
    b.view zero (physicalWrite m (b.address a) value)=(b.view zero m).write a value := by
  funext p
  have live:=b.writable_live a.region writeOnly
  unfold Binding.view physicalWrite Heap.write
  by_cases hp : p.Valid c
  · by_cases hz : b.zeroInput p.region=Bool.false
    · have he : b.address p=b.address a ↔ p=a :=
        ⟨fun h=>b.injective p a hp ha (Or.inr writeOnly) h,fun h=>congrArg b.address h⟩
      simp [hp,hz,he]
    · have hn : p≠a := by intro h; subst p; exact hz live
      simp [hp,hz,hn]
  · have hn : p≠a := fun h=>hp (h ▸ ha)
    simp [hp,hn]

namespace Source
/-- Physical stores evaluate the next RHS using current physical memory. -/
def executePhysical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (p : Phase) : List Address → Memory α → Memory α
  | [],m => m
  | a::rest,m => executePhysical ops c b scale tree p rest
      (physicalWrite m (b.address a) (rhs ops c scale tree p (b.view ops.zero m) a))

theorem execute_physical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (p : Phase) (events : List Address)
    (valid : ∀ a∈events,a.Valid c)
    (writeOnly : ∀ a∈events,a.region∉inputRegions) (entry : Memory α) :
    b.view ops.zero (executePhysical ops c b scale tree p events entry)=
      execute ops c scale tree p events (b.view ops.zero entry) := by
  induction events generalizing entry with
  | nil => rfl
  | cons a rest ih =>
    rw [executePhysical,ih (fun x hx=>valid x (by simp [hx])) (fun x hx=>writeOnly x (by simp [hx])),
      b.view_write ops.zero entry a (valid a (by simp)) (writeOnly a (by simp)),execute]

theorem physical_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (p : Phase) (events : List Address) (entry : Memory α)
    (point : Nat) (outside : ∀ a∈events,point≠b.address a) :
    executePhysical ops c b scale tree p events entry point=entry point := by
  induction events generalizing entry with
  | nil => rfl
  | cons a rest ih =>
    rw [executePhysical,ih _ (fun x hx=>outside x (by simp [hx]))]
    simp [physicalWrite,outside a (by simp)]

/-- Exact masks of a source block store imply logical cell bounds. -/
theorem tile_bounds (tag : Region) (h s rb cb nr nc rows cols : Nat) (a : Address)
    (ha : a∈tile tag h s rb cb nr nc rows cols) :
    a.region=tag ∧ a.head=h ∧ a.slice=s ∧ a.row<rows ∧ a.col<cols := by
  simp only [tile,List.mem_flatMap,List.mem_range,List.mem_filterMap] at ha
  rcases ha with ⟨i,hi,j,hj,eq⟩
  split_ifs at eq with hab
  · cases Option.some.inj eq
    exact ⟨rfl,rfl,rfl,hab.1,hab.2⟩

theorem tile_valid (c : Shape) (tag : Region) (h s rb cb nr nc rows cols : Nat)
    (validShape : ∀ i j,i<rows → j<cols → (⟨tag,h,s,i,j⟩:Address).Valid c)
    (a : Address) (ha : a∈tile tag h s rb cb nr nc rows cols) : a.Valid c := by
  rcases tile_bounds tag h s rb cb nr nc rows cols a ha with ⟨hr,hh,hs,hi,hj⟩
  rcases a with ⟨r,h',s',i,j⟩
  simp only at hr hh hs
  subst r
  subst h'
  subst s'
  exact validShape i j hi hj

theorem h_valid (c : Shape) (flag : Bool) (a : Address) (ha : a∈eventsH c flag) : a.Valid c := by
  simp only [eventsH,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,vb,hv,kb,hk,ha⟩
  rcases List.mem_append.mp ha with ha|ha
  · rcases List.mem_flatMap.mp ha with ⟨t,ht,ha⟩
    exact tile_valid c .history h t _ _ _ _ _ _ (fun i j hi hj=>⟨hh,List.mem_range.mp ht,hi,hj⟩) a ha
  · cases flag <;> simp only [Bool.false_eq_true,↓reduceIte,List.not_mem_nil] at ha
    exact tile_valid c .finalState h 0 _ _ _ _ _ _ (fun i j hi hj=>⟨hh,rfl,hi,hj⟩) a ha

theorem o_valid (c : Shape) (a : Address) (ha : a∈eventsO c) : a.Valid c := by
  simp only [eventsO,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,t,ht,vb,hv,ha⟩
  exact tile_valid c .output h 0 _ _ _ _ _ _ (fun i j hi hj=>⟨hh,rfl,hi,hj⟩) a ha

theorem dh_valid (c : Shape) (a : Address) (ha : a∈eventsDH c) : a.Valid c := by
  simp only [eventsDH,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,vb,hv,kb,hk,t,ht,ha⟩
  exact tile_valid c .dhistory h (c.chunks-1-t) _ _ _ _ _ _ (fun i j hi hj=>⟨hh,by change c.chunks-1-t<c.chunks; omega,hi,hj⟩) a ha

theorem initial_valid (c : Shape) (a : Address) (ha : a∈eventsInitial c) : a.Valid c := by
  simp only [eventsInitial,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,vb,hv,kb,hk,ha⟩
  exact tile_valid c .initialGradient h 0 _ _ _ _ _ _ (fun i j hi hj=>⟨hh,rfl,hi,hj⟩) a ha

theorem qkv_valid (c : Shape) (a : Address) (ha : a∈eventsQKV c) : a.Valid c := by
  simp only [eventsQKV,programs,List.mem_flatMap,List.mem_range,List.mem_append] at ha
  rcases ha with ⟨h,hh,t,ht,kb,hk,ha⟩
  rcases ha with (ha|ha)|ha
  · rcases ha with ⟨vb,hv,ha⟩
    exact tile_valid c .dvPart h kb _ _ _ _ _ _ (fun i j hi hj=>⟨hh,hk,hi,hj⟩) a ha
  · exact tile_valid c .dq h 0 _ _ _ _ _ _ (fun i j hi hj=>⟨hh,rfl,hi,hj⟩) a ha
  · exact tile_valid c .dk h 0 _ _ _ _ _ _ (fun i j hi hj=>⟨hh,rfl,hi,hj⟩) a ha

theorem sum_valid (c : Shape) (a : Address) (ha : a∈eventsSum c) : a.Valid c := by
  simp only [eventsSum,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨z,hz,y,hy,h,hh,ha⟩
  exact tile_valid c .dv h 0 _ _ _ _ _ _ (fun i j hi hj=>⟨hh,rfl,hi,hj⟩) a ha

theorem dhAPI_valid (c : Shape) (flag : Bool) (a : Address)
    (ha : a∈eventsDH c ++ if flag then eventsInitial c else []) : a.Valid c := by
  rcases List.mem_append.mp ha with ha|ha
  · exact dh_valid c a ha
  · cases flag <;> simp only [Bool.false_eq_true,↓reduceIte,List.not_mem_nil] at ha
    exact initial_valid c a ha

def runPhysicalAPI (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf ri : Bool) (entry : Memory α) : Memory α :=
  let mh:=executePhysical ops c b scale tree .h (eventsH c rf) entry
  let mo:=executePhysical ops c b scale tree .o (eventsO c) mh
  let mdh:=executePhysical ops c b scale tree .dh
    (eventsDH c ++ if ri then eventsInitial c else []) mo
  let mqkv:=executePhysical ops c b scale tree .qkv (eventsQKV c) mdh
  executePhysical ops c b scale tree .sum (eventsSum c) mqkv

theorem write_not_input (p : Phase) (a : Address) (ha : a.region∈p.writes) :
    a.region∉inputRegions := fun hi=>(input_separate p a.region hi) ha

theorem run_physical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf ri : Bool) (entry : Memory α) :
    b.view ops.zero (runPhysicalAPI ops c b scale tree rf ri entry)=
      runAPI ops c scale tree rf ri (b.view ops.zero entry) := by
  unfold runPhysicalAPI runAPI afterDHInitial afterO afterH
  rw [execute_physical ops c b scale tree .sum _ (sum_valid c)
        (fun a ha=>write_not_input .sum a (sum_footprint c a ha)),
      execute_physical ops c b scale tree .qkv _ (qkv_valid c)
        (fun a ha=>write_not_input .qkv a (qkv_footprint c a ha)),
      execute_physical ops c b scale tree .dh _ (dhAPI_valid c ri)
        (fun a ha=>write_not_input .dh a (initial_events_writes c ri a ha)),
      execute_physical ops c b scale tree .o _ (o_valid c)
        (fun a ha=>write_not_input .o a (o_footprint c a ha)),
      execute_physical ops c b scale tree .h _ (h_valid c rf)
        (fun a ha=>write_not_input .h a (h_footprint c rf a ha))]

theorem run_physical_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf ri : Bool) (entry : Memory α) (point : Nat)
    (outside : ∀ a∈footprint c rf ++ if ri then eventsInitial c else [],point≠b.address a) :
    runPhysicalAPI ops c b scale tree rf ri entry point=entry point := by
  unfold runPhysicalAPI
  rw [physical_frame ops c b scale tree .sum _ _ point (fun a ha=>outside a (by simp [footprint,ha])),
    physical_frame ops c b scale tree .qkv _ _ point (fun a ha=>outside a (by simp [footprint,ha])),
    physical_frame ops c b scale tree .dh _ _ point (fun a ha=>outside a (by
      rcases List.mem_append.mp ha with ha|ha
      · simp [footprint,ha]
      · exact List.mem_append_right _ ha)),
    physical_frame ops c b scale tree .o _ _ point (fun a ha=>outside a (by simp [footprint,ha])),
    physical_frame ops c b scale tree .h _ _ point (fun a ha=>outside a (by simp [footprint,ha]))]

#axiomsClean run_physical
#axiomsClean run_physical_frame
#axiomsClean execute_physical
#axiomsClean physical_frame
#axiomsClean h_valid
#axiomsClean dh_valid
#axiomsClean qkv_valid
end Source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

