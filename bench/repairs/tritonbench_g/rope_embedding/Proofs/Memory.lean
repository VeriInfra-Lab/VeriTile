import bench.repairs.tritonbench_g.rope_embedding.Proofs.Bridge
namespace VeriTile.Bench.Repairs.RopeEmbedding.Memory
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

def events {α : Type} [Inhabited α] (project : α→β) (op : Operations α) (r : Request)
    (backward : Bool) (s : State α) (base : Nat) (xs : List (Location r)) : List (Write β) :=
  xs.filterMap fun l => if active l then
    some ⟨base+4*r.offset (row l) (head l) (column l),project (sourceValue op r backward s l)⟩ else none

def run {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (backward : Bool) (s : State α) (base : Nat) (m : Store) : Store :=
  runWrites api (events project op r backward s base (schedule r)) m

theorem mem_events {α : Type} [Inhabited α] (project : α→β) (op : Operations α) (r : Request)
    (backward : Bool) (s : State α) (base : Nat) (xs : List (Location r)) (e : Write β) :
    e∈events project op r backward s base xs ↔ ∃ l∈xs,active l ∧
      e=⟨base+4*r.offset (row l) (head l) (column l),project (sourceValue op r backward s l)⟩ := by
  simp only [events,List.mem_filterMap]
  constructor
  · rintro ⟨l,hl,he⟩
    split at he
    · exact ⟨l,hl,‹_›,(Option.some.inj he).symm⟩
    · simp at he
  · rintro ⟨l,hl,ha,rfl⟩
    exact ⟨l,hl,by simp [ha]⟩

theorem readback {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (backward : Bool) (s : State α) (base : Nat) (m : Store)
    (a h c : Nat) (ha : a<r.rows) (hh : h<r.heads) (hc : c<2*r.half) :
    api.read (run api project op r backward s base m) (base+4*r.offset a h c)=
      project (scalarResult op r backward (fun o => (s.mem "Q" o).read .real)
        (fun o => (s.mem "C" o).read .real) (fun o => (s.mem "S" o).read .real) a h c) := by
  apply writes_readback
  · obtain ⟨l,hl,hal,ha',hh',hc'⟩ := coverage r a h c ha hh hc
    refine ⟨⟨base+4*r.offset (row l) (head l) (column l),project (sourceValue op r backward s l)⟩,
      (mem_events ..).mpr ⟨l,hl,hal,rfl⟩,?_,?_⟩
    · simp only [ha',hh',hc']
    · rw [source_value op r backward s l hal,ha',hh',hc']
  · intro e he
    obtain ⟨l,hl,hal,rfl⟩ := (mem_events ..).mp he
    have hcol := active_column r l hal
    have hcD : c<r.dimension := by
      have : 2*r.half≤r.dimension := by unfold Request.half; omega
      omega
    by_cases eq : r.offset (row l) (head l) (column l)=r.offset a h c
    · obtain ⟨e1,e2,e3⟩ := offset_injective r hal.2.1 hcol hh hcD eq
      left
      constructor
      · simp only [eq]
      · change project (sourceValue op r backward s l)=_
        rw [source_value op r backward s l hal,e1,e2,e3]
    · right
      dsimp only
      omega

theorem tail_readback {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (backward : Bool) (s : State α) (base : Nat) (m : Store)
    (a h c : Nat) (hh : h<r.heads) (hc : c<r.dimension) (ht : 2*r.half≤c) :
    api.read (run api project op r backward s base m) (base+4*r.offset a h c)=
      api.read m (base+4*r.offset a h c) := by
  apply writes_read_frame
  intro e he
  obtain ⟨l,hl,hal,rfl⟩ := (mem_events ..).mp he
  have ne : r.offset (row l) (head l) (column l)≠r.offset a h c := by
    intro eq
    have ec := (offset_injective r hal.2.1 (active_column r l hal) hh hc eq).2.2
    have hj := hal.2.2
    unfold column at ec
    split at ec <;> omega
  dsimp only
  omega

theorem byte_frame {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (backward : Bool) (s : State α) (base : Nat) (m : Store)
    (b : Nat) (hb : b<base ∨ base+4*r.size≤b) :
    api.byte (run api project op r backward s base m) b=api.byte m b := by
  apply writes_byte_frame
  intro e he
  obtain ⟨l,hl,ha,rfl⟩ := (mem_events ..).mp he
  have ho := offset_lt r _ _ _ ha.1 ha.2.1 (active_column r l ha)
  dsimp only
  omega

theorem active_window {α : Type} [Inhabited α] (project : α→β) (op : Operations α)
    (r : Request) (backward : Bool) (s : State α) (base : Nat) (xs : List (Location r)) (e : Write β)
    (he : e∈events project op r backward s base xs) :
    base≤e.address ∧ e.address+4≤base+4*r.size := by
  obtain ⟨l,hl,ha,rfl⟩ := (mem_events ..).mp he
  have ho := offset_lt r _ _ _ ha.1 ha.2.1 (active_column r l ha)
  dsimp only
  omega

/-- Any completed other programs preserve both Q halves of the next program.
This derives independence from packed addresses and the four-head partition. -/
theorem prefix_current {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β)
    (op : Operations α) (r : Request) (backward : Bool) (s : State α) (base : Nat) (m : Store)
    (xs : List (Location r)) (a g h c : Nat) (hh : h<r.heads) (hc : c<r.dimension)
    (hg : h/4=g) (scheduled : ∀ l∈xs,l∈schedule r)
    (other : ∀ l∈xs,row l≠a ∨ group l≠g) :
    api.read (runWrites api (events project op r backward s base xs) m) (base+4*r.offset a h c)=
      api.read m (base+4*r.offset a h c) := by
  apply writes_read_frame
  intro e he
  obtain ⟨l,hl,ha,rfl⟩ := (mem_events ..).mp he
  have ne : r.offset (row l) (head l) (column l)≠r.offset a h c := by
    intro eq
    obtain ⟨e1,e2,e3⟩ := offset_injective r ha.2.1 (active_column r l ha) hh hc eq
    have eg := group_coordinate r l (scheduled l hl)
    rcases other l hl with hn | hn
    · exact hn e1
    · exact hn (by omega)
  dsimp only
  omega

theorem no_launch_events {α : Type} [Inhabited α] (project : α→β) (op : Operations α)
    (r : Request) (backward : Bool) (s : State α) (base : Nat)
    (h : r.size=0 ∨ r.half=0) : events project op r backward s base (schedule r)=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro e he
  obtain ⟨l,hl,ha,he⟩ := (mem_events ..).mp he
  have ho := offset_lt r _ _ _ ha.1 ha.2.1 (active_column r l ha)
  rcases h with h | h
  · omega
  · have hj := ha.2.2; omega

#axiomsClean readback
#axiomsClean tail_readback
#axiomsClean byte_frame
#axiomsClean prefix_current
#axiomsClean no_launch_events
end
end VeriTile.Bench.Repairs.RopeEmbedding.Memory
