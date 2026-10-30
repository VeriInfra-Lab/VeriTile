import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000

abbrev Byte := BitVec 8
abbrev Word := BitVec 32
/-- Bytes in the checked owner's address space. -/
abbrev Memory := Nat → Byte

def encode (word : Word) (i : Nat) : Byte := word.extractLsb' (8*i) 8
def decode (bytes : Fin 4 → Byte) : Word := ((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0

theorem decode_encode (word : Word) : decode (fun i=>encode word i.val)=word := by
  change (((word.extractLsb' 24 8 ++ word.extractLsb' 16 8) ++
    word.extractLsb' 8 8) ++ word.extractLsb' 0 8)=word
  bv_decide

theorem encode_decode (bytes : Fin 4 → Byte) (i : Fin 4) : encode (decode bytes) i.val=bytes i := by
  fin_cases i
  all_goals
    apply BitVec.eq_of_getLsbD_eq
    intro bit
    simp only [encode,decode,BitVec.getLsbD_extractLsb',BitVec.getLsbD_append]
    by_cases small : bit<8
    · interval_cases bit <;> simp [BitVec.getElem_append,BitVec.getElem_extractLsb']
    · simp [BitVec.getLsbD_of_ge,show 8≤bit by omega]

def read (memory : Memory) (address : Nat) : Word := decode (fun i=>memory (address+i.val))
def words (memory : Memory) : Nat → Word := fun i=>read memory (4*i)

/-- One scalar FP32 store, with its exact four-byte exterior frame. -/
def store (memory : Memory) (address : Nat) (word : Word) : Memory := fun p=>
  if address≤p ∧ p<address+4 then encode word (p-address) else memory p

theorem store_byte (memory : Memory) (address : Nat) (word : Word) (j : Fin 4) :
    store memory address word (address+j.val)=encode word j.val := by
  simp [store,show address≤address+j.val ∧ address+j.val<address+4 by omega]

theorem store_read (memory : Memory) (address : Nat) (word : Word) :
    read (store memory address word) address=word := by
  unfold read
  simp only [store_byte]
  exact decode_encode word

theorem store_frame (memory : Memory) (address : Nat) (word : Word) (p : Nat)
    (outside : p<address ∨ address+4≤p) : store memory address word p=memory p := by
  simp [store,show ¬(address≤p ∧ p<address+4) by omega]

/-- Complete output bytes, retaining arbitrary caller bytes outside the fresh
allocation. This representation is proved equal to sequential scalar stores. -/
def render (memory : Memory) (base count : Nat) (values : Nat → Word) : Memory := fun p=>
  if base≤p ∧ p<base+4*count then encode (values ((p-base)/4)) ((p-base)%4) else memory p

def stores (memory : Memory) (base count : Nat) (values : Nat → Word) : Memory :=
  (List.range count).foldl (fun state i=>store state (base+4*i) (values i)) memory

theorem render_byte (memory : Memory) (base count : Nat) (values : Nat → Word)
    (i : Nat) (hi : i<count) (j : Fin 4) :
    render memory base count values (base+4*i+j.val)=encode (values i) j.val := by
  have inside : base≤base+4*i+j.val ∧ base+4*i+j.val<base+4*count := by omega
  have div : (base+4*i+j.val-base)/4=i := by omega
  have mod : (base+4*i+j.val-base)%4=j.val := by omega
  simp only [render,if_pos inside,div,mod]

theorem render_read (memory : Memory) (base count : Nat) (values : Nat → Word)
    (i : Nat) (hi : i<count) : read (render memory base count values) (base+4*i)=values i := by
  unfold read
  simp only [render_byte memory base count values i hi]
  exact decode_encode (values i)

theorem render_frame (memory : Memory) (base count : Nat) (values : Nat → Word)
    (p : Nat) (outside : p<base ∨ base+4*count≤p) : render memory base count values p=memory p := by
  simp [render,show ¬(base≤p ∧ p<base+4*count) by omega]

theorem render_succ (memory : Memory) (base count : Nat) (values : Nat → Word) :
    render memory base (count+1) values=store (render memory base count values) (base+4*count) (values count) := by
  funext p
  by_cases last : base+4*count≤p ∧ p<base+4*count+4
  · have inside : base≤p ∧ p<base+4*(count+1) := by omega
    have div : (p-base)/4=count := by omega
    have mod : (p-base)%4=p-(base+4*count) := by omega
    simp only [render,store,if_pos last,if_pos inside,div,mod]
  · have same : (base≤p ∧ p<base+4*(count+1)) ↔ (base≤p ∧ p<base+4*count) := by omega
    simp only [render,store,if_neg last,same]

theorem stores_render (memory : Memory) (base count : Nat) (values : Nat → Word) :
    stores memory base count values=render memory base count values := by
  induction count with
  | zero=>funext p; simp [stores,render]
  | succ n ih=>
    rw [stores,List.range_succ,List.foldl_append]
    change store (stores memory base n values) (base+4*n) (values n)=_
    rw [ih,render_succ]

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean store_read
#axiomsClean stores_render
#axiomsClean render_read
#axiomsClean render_frame
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory
