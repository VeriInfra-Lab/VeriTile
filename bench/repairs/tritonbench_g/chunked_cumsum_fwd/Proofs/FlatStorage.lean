import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Storage

/-! The ordinary same-call allocator relation and its flat-byte representation.
Allocation identities may alias within one allocation. Distinct live allocation
identities cannot overlap at any existing byte on the same device. No values,
kernel results, successful executions or frame conclusions are supplied by it. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Storage
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd RawBoundary
set_option autoImplicit false
noncomputable section
open scoped Classical

abbrev FlatHeap := Nat → Nat → BitVec 8
abbrev ByteHeap := Nat → Nat → BitVec 8

def Separated (allocation : Nat → Allocation) (live : Nat → Prop) : Prop :=
  ∀ a b i j, live a → live b → i < (allocation a).bytes → j < (allocation b).bytes →
    (allocation a).device = (allocation b).device →
    (allocation a).address + i = (allocation b).address + j → a = b

def RepresentsBytes (allocation : Nat → Allocation) (live : Nat → Prop)
    (memory : ByteHeap) (flat : FlatHeap) : Prop :=
  ∀ a q, live a → q < (allocation a).bytes →
    flat (allocation a).device ((allocation a).address + q) = memory a q

def Represents (allocation : Nat → Allocation) (live : Nat → Prop)
    (memory : PhysicalMemory Word) (flat : FlatHeap) : Prop :=
  RepresentsBytes allocation live (bytes memory) flat

def At (allocation : Nat → Allocation) (live : Nat → Prop) (device address a : Nat) : Prop :=
  live a ∧ (allocation a).device = device ∧ (allocation a).address ≤ address ∧
    address < (allocation a).address + (allocation a).bytes

/-- Realize allocation bytes on their actual device/address ranges. Addresses
outside those live ranges retain the incoming physical bytes. -/
def realizeBytes (allocation : Nat → Allocation) (live : Nat → Prop)
    (memory : ByteHeap) (fallback : FlatHeap) (device address : Nat) : BitVec 8 :=
  if h : ∃ a, At allocation live device address a then
    memory h.choose (address - (allocation h.choose).address)
  else fallback device address

def realize (allocation : Nat → Allocation) (live : Nat → Prop)
    (memory : PhysicalMemory Word) (fallback : FlatHeap) : FlatHeap :=
  realizeBytes allocation live (bytes memory) fallback

theorem realize_represents_bytes (allocation : Nat → Allocation) (live : Nat → Prop)
    (separated : Separated allocation live) (memory : ByteHeap) (fallback : FlatHeap) :
    RepresentsBytes allocation live memory (realizeBytes allocation live memory fallback) := by
  intro a q ha hq
  have found : ∃ b, At allocation live (allocation a).device ((allocation a).address+q) b :=
    ⟨a,ha,rfl,by omega,by omega⟩
  rw [realizeBytes,dif_pos found]
  have chosen := found.choose_spec
  have owner : found.choose = a := separated found.choose a
    ((allocation a).address+q-(allocation found.choose).address) q chosen.1 ha
    (by have := chosen.2.2; omega) hq chosen.2.1 (by have := chosen.2.2; omega)
  simp [owner]

theorem realize_represents (allocation : Nat → Allocation) (live : Nat → Prop)
    (separated : Separated allocation live) (memory : PhysicalMemory Word) (fallback : FlatHeap) :
    Represents allocation live memory (realize allocation live memory fallback) :=
  realize_represents_bytes allocation live separated (bytes memory) fallback

/-- Reifying a sequence of source-computed heaps retains the same native
addresses. It equals reifying the final heap with the original exterior. -/
theorem realize_sequence (allocation : Nat → Allocation) (live : Nat → Prop)
    (before after : PhysicalMemory Word) (flat : FlatHeap) :
    realize allocation live after (realize allocation live before flat) =
      realize allocation live after flat := by
  funext device address
  unfold realize realizeBytes
  split_ifs <;> rfl

/-- Representation is inhabited for every abstract word memory; it is not a
postulated whole-output property. -/
theorem representation_exists (allocation : Nat → Allocation) (live : Nat → Prop)
    (separated : Separated allocation live) (memory : PhysicalMemory Word) :
    ∃ flat, Represents allocation live memory flat :=
  ⟨realize allocation live memory (fun _ _ => 0),
    realize_represents allocation live separated memory (fun _ _ => 0)⟩

theorem represented_native_bytes {allocation : Nat → Allocation} {live : Nat → Prop}
    {memory : PhysicalMemory Word} {flat : FlatHeap}
    (representation : Represents allocation live memory flat) (t : Tensor allocation)
    (resident : live t.view.allocation) (off : Nat) (covered : t.Covers off)
    (i : Nat) (hi : i < 4) :
    flat t.device ((Index.address t.dataPtr (Index.u off)).toNat+i) =
      byte (t.view.read memory off) i := by
  rw [native_address t off covered,t.sameDevice,t.pointer off covered]
  have bound : 4*(t.view.base+off)+i < (allocation t.view.allocation).bytes := by
    unfold Tensor.Covers at covered
    omega
  have rep := representation t.view.allocation (4*(t.view.base+off)+i) resident bound
  convert rep.trans (read_bytes memory t off i hi) using 1 <;> congr 1 <;> omega

theorem represented_frame {allocation : Nat → Allocation} {live old : Nat → Prop}
    {before after : PhysicalMemory Word} {entry final : FlatHeap}
    (initial : Represents allocation old before entry)
    (result : Represents allocation live after final)
    (kept : ∀ a, old a → live a)
    (frame : ∀ a q, old a → after a q = before a q)
    (a q : Nat) (ha : old a) (hq : q < (allocation a).bytes) :
    final (allocation a).device ((allocation a).address+q) =
      entry (allocation a).device ((allocation a).address+q) := by
  rw [result a q (kept a ha) hq,initial a q ha hq]
  exact all_byte_frame before after old frame a q ha

/-- Reification changes only physical images of abstractly written words.
This applies to every trial, selected launch and any source prefix whose
ordinary store support has been derived. Partial trailing bytes are included. -/
theorem realize_outside {allocation : Nat → Allocation} {live : Nat → Prop}
    {before after : PhysicalMemory Word} {flat : FlatHeap} {written : Nat → Nat → Prop}
    (representation : Represents allocation live before flat)
    (frame : ∀ a q, ¬ written a q → after a q = before a q)
    (device address : Nat)
    (outside : ¬ ∃ a q, live a ∧ q < (allocation a).bytes ∧ written a (q/4) ∧
      (allocation a).device = device ∧ (allocation a).address+q = address) :
    realize allocation live after flat device address = flat device address := by
  unfold realize realizeBytes
  split_ifs with h
  · have hc := h.choose_spec
    let a := h.choose
    let q := address - (allocation a).address
    have hq : q < (allocation a).bytes := by have := hc.2.2; dsimp [q,a]; omega
    have addr : (allocation a).address+q = address := by have := hc.2.2; dsimp [q,a]; omega
    have unwritten : ¬ written a (q/4) := fun hw => outside ⟨a,q,hc.1,hq,hw,hc.2.1,addr⟩
    have rep := representation a q hc.1 hq
    rw [hc.2.1,addr] at rep
    change bytes after a q = _
    rw [bytes,frame a (q/4) unwritten]
    exact rep.symm
  · rfl

/-- A byte from any actual word store, including an interrupted store. The
payload is a local source value; the allocator relation constrains no payload. -/
structure ByteWrite (allocation : Nat → Allocation) (live : Nat → Prop) where
  owner : Nat
  offset : Nat
  value : BitVec 8
  resident : live owner
  bounded : offset < (allocation owner).bytes

def ByteWrite.abstract {allocation : Nat → Allocation} {live : Nat → Prop}
    (e : ByteWrite allocation live) (memory : ByteHeap) : ByteHeap :=
  fun a q => if a = e.owner ∧ q = e.offset then e.value else memory a q

def ByteWrite.native {allocation : Nat → Allocation} {live : Nat → Prop}
    (e : ByteWrite allocation live) (flat : FlatHeap) : FlatHeap :=
  fun device address => if device = (allocation e.owner).device ∧
      address = (allocation e.owner).address+e.offset then e.value else flat device address

theorem ByteWrite.simulation {allocation : Nat → Allocation} {live : Nat → Prop}
    (separated : Separated allocation live) (e : ByteWrite allocation live)
    {memory : ByteHeap} {flat : FlatHeap} (rep : RepresentsBytes allocation live memory flat) :
    RepresentsBytes allocation live (e.abstract memory) (e.native flat) := by
  intro a q ha hq
  have same : ((allocation a).device = (allocation e.owner).device ∧
      (allocation a).address+q = (allocation e.owner).address+e.offset) ↔
      (a = e.owner ∧ q = e.offset) := by
    constructor
    · rintro ⟨hd,hp⟩
      have eq := separated a e.owner q e.offset ha e.resident hq e.bounded hd hp
      exact ⟨eq,by simpa only [eq,Nat.add_left_cancel_iff] using hp⟩
    · rintro ⟨rfl,rfl⟩; exact ⟨rfl,rfl⟩
  simp only [ByteWrite.native,ByteWrite.abstract,same]
  split_ifs <;> first | rfl | exact rep a q ha hq

theorem byte_trace_simulation {allocation : Nat → Allocation} {live : Nat → Prop}
    (separated : Separated allocation live) (events : List (ByteWrite allocation live))
    {memory : ByteHeap} {flat : FlatHeap} (rep : RepresentsBytes allocation live memory flat) :
    RepresentsBytes allocation live
      (events.foldl (fun m e => e.abstract m) memory)
      (events.foldl (fun m e => e.native m) flat) := by
  induction events generalizing memory flat with
  | nil => exact rep
  | cons e es ih => exact ih (e.simulation separated rep)

theorem byte_trace_frame {allocation : Nat → Allocation} {live old : Nat → Prop}
    (events : List (ByteWrite allocation live))
    (fresh : ∀ e ∈ events, ¬ old e.owner) (memory : ByteHeap)
    (a q : Nat) (ha : old a) :
    events.foldl (fun m e => e.abstract m) memory a q = memory a q := by
  induction events generalizing memory with
  | nil => rfl
  | cons e es ih =>
    rw [List.foldl_cons,ih (fun x hx => fresh x (List.mem_cons_of_mem e hx))]
    have different : a ≠ e.owner := fun eq => fresh e (List.mem_cons_self) (eq ▸ ha)
    simp [ByteWrite.abstract,different]

/-- Every interrupted prefix has the actual native caller-byte frame. This
also covers non-word-aligned caller tails and aliases of a caller allocation. -/
theorem byte_trace_native_prefix_frame {allocation : Nat → Allocation} {live old : Nat → Prop}
    (separated : Separated allocation live) (events : List (ByteWrite allocation live))
    (fresh : ∀ e ∈ events, ¬ old e.owner) (kept : ∀ a, old a → live a)
    {memory : ByteHeap} {flat : FlatHeap} (rep : RepresentsBytes allocation live memory flat)
    (n a q : Nat) (ha : old a) (hq : q < (allocation a).bytes) :
    ((events.take n).foldl (fun m e => e.native m) flat)
      (allocation a).device ((allocation a).address+q) =
        flat (allocation a).device ((allocation a).address+q) := by
  have after := byte_trace_simulation separated (events.take n) rep
  rw [after a q (kept a ha) hq,
    byte_trace_frame (events.take n) (fun e he => fresh e (List.mem_of_mem_take he)) memory a q ha,
    rep a q (kept a ha) hq]

theorem separated_rejects_collision {allocation : Nat → Allocation} {live : Nat → Prop}
    (a b : Nat) (ha : live a) (hb : live b) (different : a ≠ b)
    (apos : 0 < (allocation a).bytes) (bpos : 0 < (allocation b).bytes)
    (device : (allocation a).device = (allocation b).device)
    (address : (allocation a).address = (allocation b).address) :
    ¬ Separated allocation live := by
  intro separated
  exact different (separated a b 0 0 ha hb apos bpos device (by simpa using address))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Storage
