import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Actual byte-addressed data memory for dereferenced words. A nonempty view
starts at its data pointer, including a possible nonzero storage offset. For
an empty view the affine storage coordinate is unused and may differ from
Torch's reported null data_ptr. Bounds prevent every empty-view dereference.
Frames include all allocation padding. Allocator bookkeeping is not modeled. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory

abbrev Byte := Fin 256
abbrev Heap := Nat → Byte

/-- Ordinary four-byte scalar representation interface; neither floating
arithmetic nor task output values are assumed by this interface. -/
structure Codec (Word : Type) where
  encode : Word → Fin 4 → Byte
  decode : (Fin 4 → Byte) → Word
  roundtrip : ∀ w, decode (encode w) = w

def load {Word : Type} (codec : Codec Word) (m : Heap) (address : Nat) : Word :=
  codec.decode (fun k => m (address + k.val))

def store {Word : Type} (codec : Codec Word) (m : Heap) (address : Nat) (w : Word) : Heap :=
  fun p => if h : address ≤ p ∧ p < address + 4 then
    codec.encode w ⟨p - address, by omega⟩ else m p

theorem store_byte {Word : Type} (codec : Codec Word) (m : Heap)
    (address : Nat) (w : Word) (k : Fin 4) :
    store codec m address w (address + k.val) = codec.encode w k := by
  simp only [store]
  rw [dif_pos (by omega)]
  congr 1
  apply Fin.ext
  dsimp only
  omega

theorem load_store {Word : Type} (codec : Codec Word) (m : Heap) (address : Nat) (w : Word) :
    load codec (store codec m address w) address = w := by
  unfold load
  have h : (fun k : Fin 4 => store codec m address w (address + k.val)) = codec.encode w := by
    funext k
    exact store_byte codec m address w k
  rw [h, codec.roundtrip]

theorem store_frame {Word : Type} (codec : Codec Word) (m : Heap)
    (address : Nat) (w : Word) (p : Nat) (outside : p < address ∨ address + 4 ≤ p) :
    store codec m address w p = m p := by
  unfold store
  rw [dif_neg (by omega)]

theorem load_store_other {Word : Type} (codec : Codec Word) (m : Heap)
    (address other : Nat) (w : Word) (apart : other + 4 ≤ address ∨ address + 4 ≤ other) :
    load codec (store codec m address w) other = load codec m other := by
  unfold load
  congr 1
  funext k
  apply store_frame
  omega

structure Backing where
  base : Nat
  bytes : Nat
  native : base + bytes ≤ 2 ^ 64

def Backing.contains (b : Backing) (p : Nat) := b.base ≤ p ∧ p < b.base + b.bytes

def Backing.disjoint (a b : Backing) := a.base + a.bytes ≤ b.base ∨ b.base + b.bytes ≤ a.base

theorem Backing.disjoint_symm (a b : Backing) (h : a.disjoint b) : b.disjoint a := by
  exact h.symm

structure View where
  backing : Backing
  /-- Offset in four-byte scalar words, including complex interleaving. -/
  offset : Nat
  words : Nat
  capacity : 4 * (offset + words) ≤ backing.bytes

def View.pointer (v : View) := v.backing.base + 4 * v.offset
def View.address (v : View) (i : Nat) := v.pointer + 4 * i

theorem View.address_offset (v : View) (i : Nat) :
    v.address i = v.backing.base + 4 * (v.offset + i) := by
  simp [View.address, View.pointer]
  ring

theorem View.byte_inside (v : View) (i : Nat) (hi : i < v.words) (k : Fin 4) :
    v.backing.contains (v.address i + k.val) := by
  have hc := v.capacity
  simp only [Backing.contains, View.address, View.pointer]
  constructor <;> omega

theorem View.address_native (v : View) (i : Nat) (hi : i < v.words) :
    v.address i + 4 ≤ 2 ^ 64 := by
  have hc := v.capacity
  have hn := v.backing.native
  simp only [View.address, View.pointer]
  omega

theorem View.byte_count_lt (v : View) (nonnull : 0 < v.words → 0 < v.backing.base) :
    4 * v.words < 2 ^ 64 := by
  have hc := v.capacity
  have hn := v.backing.native
  by_cases hz : v.words = 0
  · simp [hz]
  · have hb := nonnull (by omega)
    omega

theorem View.word_disjoint (v : View) (i j : Nat) (hne : i ≠ j) :
    v.address i + 4 ≤ v.address j ∨ v.address j + 4 ≤ v.address i := by
  simp only [View.address]
  omega

theorem store_other_backing {Word : Type} (codec : Codec Word) (m : Heap)
    (output : View) (caller : Backing) (fresh : output.backing.disjoint caller)
    (i : Nat) (hi : i < output.words) (w : Word) (p : Nat) (hp : caller.contains p) :
    store codec m (output.address i) w p = m p := by
  apply store_frame
  have hc := output.capacity
  simp only [Backing.disjoint] at fresh
  simp only [Backing.contains] at hp
  simp only [View.address, View.pointer]
  omega

def Frame (before after : Heap) (caller : Backing) : Prop :=
  ∀ p, caller.contains p → after p = before p

theorem Frame.refl (m : Heap) (caller : Backing) : Frame m m caller := by
  intro p _
  rfl

theorem Frame.trans {a b c : Heap} {caller : Backing}
    (hab : Frame a b caller) (hbc : Frame b c caller) : Frame a c caller := by
  intro p hp
  rw [hbc p hp, hab p hp]

/-- A scalar-store trace. Iteration order is retained, including repeated
stores; nothing is assumed about the values in the trace. -/
def stores {Word : Type} (codec : Codec Word) (events : List (Nat × Word)) (m : Heap) : Heap :=
  events.foldl (fun current event => store codec current event.1 event.2) m

theorem stores_frame {Word : Type} (codec : Codec Word) (events : List (Nat × Word))
    (m : Heap) (p : Nat)
    (outside : ∀ event ∈ events, p < event.1 ∨ event.1 + 4 ≤ p) :
    stores codec events m p = m p := by
  induction events generalizing m with
  | nil => rfl
  | cons event rest ih =>
    change stores codec rest (store codec m event.1 event.2) p = m p
    rw [ih _ (by intro e he; exact outside e (by simp [he]))]
    exact store_frame _ _ _ _ _ (outside event (by simp))

theorem stores_load_other {Word : Type} (codec : Codec Word) (events : List (Nat × Word))
    (m : Heap) (address : Nat)
    (apart : ∀ event ∈ events, address + 4 ≤ event.1 ∨ event.1 + 4 ≤ address) :
    load codec (stores codec events m) address = load codec m address := by
  unfold load
  congr 1
  funext k
  apply stores_frame
  intro event he
  have h := apart event he
  omega

/-- Readback of an actual stored word, when every other store is disjoint.
This covers output ownership without assuming any initial empty_like value. -/
theorem stores_readback {Word : Type} (codec : Codec Word) (events : List (Nat × Word))
    (m : Heap) (address : Nat) (w : Word) (member : (address, w) ∈ events)
    (owners : ∀ event ∈ events, event = (address, w) ∨
      address + 4 ≤ event.1 ∨ event.1 + 4 ≤ address) :
    load codec (stores codec events m) address = w := by
  induction events generalizing m with
  | nil => simp at member
  | cons first rest ih =>
    change load codec (stores codec rest (store codec m first.1 first.2)) address = w
    by_cases hin : (address, w) ∈ rest
    · exact ih _ hin (by intro e he; exact owners e (by simp [he]))
    · have heq : first = (address, w) := ((List.mem_cons.mp member).resolve_right hin).symm
      have htail : ∀ event ∈ rest, address + 4 ≤ event.1 ∨ event.1 + 4 ≤ address := by
        intro event he
        rcases owners event (by simp [he]) with h | h
        · exact False.elim (hin (h ▸ he))
        · exact h
      rw [stores_load_other _ _ _ _ htail, heq]
      exact load_store _ _ _ _

/-- Exactly the ordered scalar stores of one computed vector/tile. -/
def writeValues {Word : Type} (codec : Codec Word) (v : View) (indices : List Nat)
    (values : Nat → Word) (m : Heap) : Heap :=
  stores codec (indices.map (fun i => (v.address i, values i))) m

theorem writeValues_read {Word : Type} (codec : Codec Word) (v : View) (indices : List Nat)
    (values : Nat → Word) (m : Heap) (i : Nat) (member : i ∈ indices) :
    load codec (writeValues codec v indices values m) (v.address i) = values i := by
  apply stores_readback
  · exact List.mem_map.mpr ⟨i, member, rfl⟩
  · intro event he
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp he
    by_cases h : j = i
    · subst j
      exact Or.inl rfl
    · exact Or.inr (v.word_disjoint i j (Ne.symm h))

theorem writeValues_frame {Word : Type} (codec : Codec Word) (v : View) (indices : List Nat)
    (values : Nat → Word) (m : Heap) (p : Nat)
    (outside : ∀ i ∈ indices, p < v.address i ∨ v.address i + 4 ≤ p) :
    writeValues codec v indices values m p = m p := by
  apply stores_frame
  intro event he
  obtain ⟨i, hi, rfl⟩ := List.mem_map.mp he
  exact outside i hi

theorem writeValues_load_outside {Word : Type} (codec : Codec Word) (v : View)
    (indices : List Nat) (values : Nat → Word) (m : Heap) (i : Nat) (outside : i ∉ indices) :
    load codec (writeValues codec v indices values m) (v.address i) = load codec m (v.address i) := by
  apply stores_load_other
  intro event he
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp he
  apply v.word_disjoint
  intro h
  exact outside (h ▸ hj)

theorem writeValues_other_backing {Word : Type} (codec : Codec Word) (v : View)
    (indices : List Nat) (values : Nat → Word) (m : Heap) (caller : Backing)
    (fresh : v.backing.disjoint caller) (bounded : ∀ i ∈ indices, i < v.words) :
    Frame m (writeValues codec v indices values m) caller := by
  intro p hp
  apply writeValues_frame
  intro i hi
  have hv := bounded i hi
  have hc := v.capacity
  simp only [Backing.contains] at hp
  simp only [Backing.disjoint] at fresh
  simp only [View.address, View.pointer]
  omega

theorem Frame.load {Word : Type} (codec : Codec Word) {before after : Heap}
    (v : View) (frame : Frame before after v.backing) (i : Nat) (hi : i < v.words) :
    load codec after (v.address i) = load codec before (v.address i) := by
  unfold ByteMemory.load
  congr 1
  funext k
  exact frame _ (v.byte_inside i hi k)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
