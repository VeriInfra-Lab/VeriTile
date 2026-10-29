import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Bridge

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Memory
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000

/-- Ordinary four-byte memory operations, with an explicit byte codec. -/
structure API (α Store Byte : Type) where
  read : Store → Nat → α
  write : Store → Nat → α → Store
  byte : Store → Nat → Byte
  encode : α → Fin 4 → Byte
  read_bytes : ∀ m a k, byte m (a+k.val) = encode (read m a) k
  read_same : ∀ m a v, read (write m a v) a = v
  read_apart : ∀ m a b v, (a+4≤b ∨ b+4≤a) → read (write m a v) b = read m b
  byte_frame : ∀ m a b v, (b<a ∨ a+4≤b) → byte (write m a v) b = byte m b

structure Write (α : Type) where
  address : Nat
  value : α

def runWrites (api : API α Store Byte) (events : List (Write α)) (m : Store) : Store :=
  events.foldl (fun state e => api.write state e.address e.value) m

theorem writes_read_frame (api : API α Store Byte) (es : List (Write α)) (m : Store) (a : Nat)
    (h : ∀ e ∈ es, e.address+4≤a ∨ a+4≤e.address) :
    api.read (runWrites api es m) a = api.read m a := by
  induction es generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.read (runWrites api es (api.write m e.address e.value)) a = _
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.read_apart _ _ _ _ (h e (List.mem_cons_self ..))

theorem writes_byte_frame (api : API α Store Byte) (es : List (Write α)) (m : Store) (b : Nat)
    (h : ∀ e ∈ es, b<e.address ∨ e.address+4≤b) :
    api.byte (runWrites api es m) b = api.byte m b := by
  induction es generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.byte (runWrites api es (api.write m e.address e.value)) b = _
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.byte_frame _ _ _ _ (h e (List.mem_cons_self ..))

theorem writes_readback (api : API α Store Byte) (es : List (Write α)) (m : Store) (a : Nat) (v : α)
    (present : ∃ e ∈ es, e.address=a ∧ e.value=v)
    (only : ∀ e ∈ es, (e.address=a ∧ e.value=v) ∨ e.address+4≤a ∨ a+4≤e.address) :
    api.read (runWrites api es m) a = v := by
  induction es generalizing m with
  | nil => simp at present
  | cons e es ih =>
    change api.read (runWrites api es (api.write m e.address e.value)) a = _
    by_cases tail : ∃ q ∈ es, q.address=a ∧ q.value=v
    · exact ih _ tail (fun q hq => only q (List.mem_cons_of_mem _ hq))
    · have head : e.address=a ∧ e.value=v := by
        obtain ⟨q,hq,ha,hv⟩ := present
        rcases List.mem_cons.mp hq with he | ht
        · simpa [he] using And.intro ha hv
        · exact (tail ⟨q,ht,ha,hv⟩).elim
      rw [writes_read_frame api es _ a]
      · rw [head.1,head.2,api.read_same]
      · intro q hq
        rcases only q (List.mem_cons_of_mem _ hq) with same | apart
        · exact (tail ⟨q,hq,same⟩).elim
        · exact apart

def value {α : Type} [Inhabited α] (ops : Operations α) (r : Request) (input output : RegionName)
    (s : State α) (l : Location) : α :=
  Source.value ops.backend (parameters r input output) (programState s l) (imaginary l) (lane l)

def events {α : Type} [Inhabited α] (project : α → β) (ops : Operations α) (r : Request) (input output : RegionName)
    (s : State α) (base : Nat) (xs : List Location) : List (Write β) :=
  xs.filterMap fun l => if active r l then
    some ⟨base+4*r.offset (batch l) (row r l) (column r l),project (value ops r input output s l)⟩ else none

def run {α : Type} [Inhabited α] (api : API β Store Byte) (project : α → β) (ops : Operations α)
    (r : Request) (input output : RegionName) (s : State α) (base : Nat) (m : Store) : Store :=
  runWrites api (events project ops r input output s base (schedule r)) m

theorem mem_events {α : Type} [Inhabited α] (project : α → β) (ops : Operations α) (r : Request) (input output : RegionName)
    (s : State α) (base : Nat) (xs : List Location) (e : Write β) :
    e ∈ events project ops r input output s base xs ↔
      ∃ l ∈ xs, active r l ∧ e = ⟨base+4*r.offset (batch l) (row r l) (column r l),project (value ops r input output s l)⟩ := by
  simp only [events,List.mem_filterMap]
  constructor
  · rintro ⟨l,hl,he⟩
    split at he
    · exact ⟨l,hl,‹_›,(Option.some.inj he).symm⟩
    · simp at he
  · rintro ⟨l,hl,ha,rfl⟩
    exact ⟨l,hl,by simp [ha]⟩

theorem readback {α : Type} [Inhabited α] (api : API β Store Byte) (project : α → β) (ops : Operations α)
    (r : Request) (input output : RegionName) (s : State α) (base : Nat) (m : Store)
    (b rowIndex c : Nat) (hb : b<r.batches) (hm : rowIndex<r.rows) (hc : c<r.columns) :
    api.read (run api project ops r input output s base m) (base+4*r.offset b rowIndex c) =
      project (scalarResult ops r (fun o => (s.mem input o).read .fp32) b rowIndex c) := by
  apply writes_readback
  · obtain ⟨l,hl,ha,hb',hm',hc'⟩ := coverage r b rowIndex c hb hm hc
    refine ⟨⟨base+4*r.offset (batch l) (row r l) (column r l),project (value ops r input output s l)⟩,
      (mem_events ..).mpr ⟨l,hl,ha,rfl⟩,?_,?_⟩
    · simp only [hb',hm',hc']
    · rw [value,source_value ops r input output s l ha,hb',hm',hc']
  · intro e he
    obtain ⟨l,hl,ha,rfl⟩ := (mem_events ..).mp he
    by_cases h : r.offset (batch l) (row r l) (column r l) = r.offset b rowIndex c
    · obtain ⟨h1,h2,h3⟩ := offset_injective r ha.2.1 ha.2.2 hm hc h
      left
      constructor
      · simp only [h]
      · change project (value ops r input output s l) = _
        rw [value,source_value ops r input output s l ha,h1,h2,h3]
    · right
      dsimp only
      omega

theorem byte_frame {α : Type} [Inhabited α] (api : API β Store Byte) (project : α → β) (ops : Operations α)
    (r : Request) (input output : RegionName) (s : State α) (base : Nat) (m : Store)
    (b : Nat) (h : b<base ∨ base+4*r.size≤b) :
    api.byte (run api project ops r input output s base m) b = api.byte m b := by
  apply writes_byte_frame
  intro e he
  obtain ⟨l,hl,ha,rfl⟩ := (mem_events ..).mp he
  have ho := offset_lt r _ _ _ ha.1 ha.2.1 ha.2.2
  dsimp only
  omega

theorem active_window {α : Type} [Inhabited α] (project : α → β) (ops : Operations α) (r : Request)
    (input output : RegionName) (s : State α) (base : Nat) (e : Write β)
    (he : e ∈ events project ops r input output s base (schedule r)) :
    base≤e.address ∧ e.address+4≤base+4*r.size := by
  obtain ⟨l,hl,ha,rfl⟩ := (mem_events ..).mp he
  have ho := offset_lt r _ _ _ ha.1 ha.2.1 ha.2.2
  dsimp only
  omega

#axiomsClean readback
#axiomsClean byte_frame
#axiomsClean active_window
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Memory
