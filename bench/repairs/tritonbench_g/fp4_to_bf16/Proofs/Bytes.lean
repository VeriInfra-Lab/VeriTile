import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Finite
import bench.tritonbench_g.fp4_to_bf16.Proofs.ScheduleCorrect

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Bytes
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostCorrect
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev Heap := Nat → BitVec 8
/-- Little-endian BF16 byte representation of a logical output cell. -/
def byte (bits : BitVec 16) (k : Nat) : BitVec 8 :=
  if k % 2 = 0 then bits.extractLsb' 0 8 else bits.extractLsb' 8 8

def read (heap : Heap) (address : Nat) : BitVec 16 :=
  heap (address+1) ++ heap address

theorem byte_roundtrip (bits : BitVec 16) :
    byte bits 1 ++ byte bits 0 = bits := by
  simp only [byte]
  bv_decide

/-- Representation of a completed logical memory in a byte-addressed allocation.
This is a representation map, not an independent numerical target or a claim
that invoking it executes a GPU. All outside bytes retain the initial heap. -/
def physical (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (memory : RegionName → Nat → Cell Word) : Heap := fun a =>
  if s.outputBase ≤ a ∧ a < s.outputBase+4*h.shape.count then
    byte (readBf16 (memory h.output ((a-s.outputBase)/2))).bits ((a-s.outputBase)%2)
  else heap a

theorem physical_byte (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (memory : RegionName → Nat → Cell Word) (j k : Nat)
    (hj : j<2*h.shape.count) (hk : k<2) :
    physical h s heap memory (s.outputBase+2*j+k) =
      byte (readBf16 (memory h.output j)).bits k := by
  have inside : s.outputBase ≤ s.outputBase+2*j+k ∧
      s.outputBase+2*j+k < s.outputBase+4*h.shape.count := by omega
  have div : (s.outputBase+2*j+k-s.outputBase)/2=j := by omega
  have mod : (s.outputBase+2*j+k-s.outputBase)%2=k := by omega
  simp only [physical, inside, and_self, if_true, div, mod]

theorem physical_read (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (memory : RegionName → Nat → Cell Word) (j : Nat) (hj : j<2*h.shape.count) :
    read (physical h s heap memory) (s.outputBase+2*j) =
      (readBf16 (memory h.output j)).bits := by
  unfold read
  rw [physical_byte h s heap memory j 1 hj (by omega)]
  have h0 := physical_byte h s heap memory j 0 hj (by omega)
  simp only [Nat.add_zero] at h0
  rw [h0, byte_roundtrip]

theorem physical_frame (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (memory : RegionName → Nat → Cell Word) (a : Nat)
    (outside : a<s.outputBase ∨ s.outputBase+4*h.shape.count≤a) :
    physical h s heap memory a = heap a := by
  have inactive : ¬ (s.outputBase≤a ∧ a<s.outputBase+4*h.shape.count) := by omega
  simp only [physical, inactive, if_false]

theorem physical_input (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (memory : RegionName → Nat → Cell Word) (i : Nat) (hi : i<h.shape.count) :
    physical h s heap memory (s.inputPointer+i) = heap (s.inputPointer+i) := by
  apply physical_frame
  have := s.inputFits
  have := s.outputFits
  have sep := s.separate
  dsimp [Finite.Storage.inputPointer]
  rcases sep with sep | sep <;> omega

/-- Initial physical input relation. It refers only to one-byte raw uint8
storage at its already-offset pointer; arbitrary initial BF16 output is allowed. -/
def InputRep (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) : Prop :=
  ∀ i, i<h.shape.count → initial.mem h.input i = .mk .nat (heap (s.inputPointer+i)).toNat

theorem packed_of_rep (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) (rep : InputRep h s heap initial) : PackedInput h initial := by
  intro i hi
  exact ⟨heap (s.inputPointer+i), rep i hi⟩

/-- Independent raw-input nibble at any logical output coordinate. -/
def targetNibble (s : Finite.Storage n) (heap : Heap) (j : Nat) : Bits.Nibble :=
  if j%2=0 then Bits.high (heap (s.inputPointer+j/2)) else Bits.low (heap (s.inputPointer+j/2))

theorem nibble_from_bytes (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) (rep : InputRep h s heap initial) (j : Nat) (hj : j<2*h.shape.count) :
    nibble h initial j = targetNibble s heap j := by
  unfold nibble targetNibble
  rw [rep (j/2) (by omega)]
  simp only [Cell.read_same, BitVec.ofNat_toNat, BitVec.setWidth_eq]

/-- Generic two-byte store, independent of the FP4 numerical reference. -/
def store (heap : Heap) (address : Nat) (bits : BitVec 16) : Heap := fun a =>
  if a=address then byte bits 0 else if a=address+1 then byte bits 1 else heap a

def update (h : Input) (memory : RegionName → Nat → Cell Word) (j : Nat)
    (bits : BitVec 16) : RegionName → Nat → Cell Word := fun r o =>
  if r=h.output ∧ o=j then .mk .bf16 (embed ⟨bits⟩) else memory r o

theorem read_embed (bits : BitVec 16) :
    (readBf16 (.mk .bf16 (embed ⟨bits⟩))).bits = bits := by
  simp only [readBf16, Cell.read_same, embed]
  simpa only [BitVec.setWidth_eq] using
    (BitVec.setWidth_setWidth_of_le bits (by decide : 16 ≤ 32))

/-- The physical representation commutes with each actual active BF16 store.
Together with input preservation and finite masks this is the primitive memory
connection, not a premise about the final kernel result. -/
theorem physical_store (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (memory : RegionName → Nat → Cell Word) (j : Nat) (hj : j<2*h.shape.count)
    (bits : BitVec 16) :
    physical h s heap (update h memory j bits) =
      store (physical h s heap memory) (s.outputBase+2*j) bits := by
  funext a
  by_cases h0 : a=s.outputBase+2*j
  · subst a
    have hb := physical_byte h s heap (update h memory j bits) j 0 hj (by omega)
    simp only [Nat.add_zero] at hb
    rw [hb]
    simp [update, read_embed, store]
  · by_cases h1 : a=s.outputBase+2*j+1
    · subst a
      rw [physical_byte h s heap (update h memory j bits) j 1 hj (by omega)]
      simp [update, read_embed, store]
    · simp only [store, h0, h1, if_false]
      unfold physical
      split_ifs with inside
      · have ne : (a-s.outputBase)/2 ≠ j := by omega
        simp only [update, ne, and_false, if_false]
      · rfl

/-- Full initial byte representation. The output condition specifies only how
arbitrary pre-launch bytes are represented, not their numerical contents. -/
structure StorageRep (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) : Prop where
  input : InputRep h s heap initial
  output : ∀ j, j<2*h.shape.count →
    (readBf16 (initial.mem h.output j)).bits = read heap (s.outputBase+2*j)

theorem byte_read (heap : Heap) (address k : Nat) (hk : k<2) :
    byte (read heap address) k=heap (address+k) := by
  have cases : k=0 ∨ k=1 := by omega
  rcases cases with rfl | rfl
  · simp only [byte, read, Nat.add_zero]
    bv_decide
  · simp only [byte, read, Nat.add_zero]
    bv_decide

theorem physical_initial (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) (rep : StorageRep h s heap initial) :
    physical h s heap initial.mem=heap := by
  funext a
  by_cases inside : s.outputBase≤a ∧ a<s.outputBase+4*h.shape.count
  · let j := (a-s.outputBase)/2
    let k := (a-s.outputBase)%2
    have hj : j<2*h.shape.count := by dsimp [j]; omega
    have hk : k<2 := by dsimp [k]; omega
    have addr : s.outputBase+2*j+k=a := by dsimp [j,k]; omega
    rw [←addr, physical_byte h s heap initial.mem j k hj hk, rep.output j hj,
      byte_read heap (s.outputBase+2*j) k hk]
  · exact physical_frame h s heap initial.mem a (by omega)

/-- Every heap, including arbitrary torch.empty output bits, has a matching
logical initial state; the representation premise cannot exclude an allocation. -/
def represent (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) : State Word :=
  { initial with mem := fun r o =>
      if r=h.input then .mk .nat (heap (s.inputPointer+o)).toNat
      else if r=h.output then .mk .bf16 (embed ⟨read heap (s.outputBase+2*o)⟩)
      else initial.mem r o }

theorem represent_correct (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) (fresh : h.input≠h.output) :
    StorageRep h s heap (represent h s heap initial) := by
  constructor
  · intro i hi
    simp only [represent, if_true]
  · intro j hj
    simp only [represent, Ne.symm fresh, if_false, if_true]
    exact read_embed _

#axiomsClean physical_initial
#axiomsClean represent_correct
#axiomsClean physical_store
#specNonCircular targetNibble avoiding [physical, Host.run, HostSchedule.runSchedule]
#axiomsClean physical_read
#axiomsClean physical_frame
#axiomsClean physical_input
#axiomsClean nibble_from_bytes
end
end VeriTile.Bench.Repairs.Fp4ToBf16.Bytes
