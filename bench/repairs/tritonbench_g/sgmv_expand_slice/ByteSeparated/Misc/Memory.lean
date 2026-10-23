import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Task-local copies of the existing generic Key/store and four-byte packing
lemmas. They avoid importing unrelated kernel proofs whose cached generated
simp equations conflict with the retained SGMV source modules. No allocation,
scheduling, compiler, separation or numerical-result assumption is added. -/
namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- An owner and the actual byte start of a four-byte FP32 span. Caller
starts may be unaligned; distinct owners may reuse numeric virtual addresses. -/
structure Key where
  owner : Nat
  address : Nat
  deriving DecidableEq

def Key.covers (key : Key) (owner address : Nat) : Prop :=
  owner=key.owner ∧ key.address≤address ∧ address<key.address+4

def Key.disjoint (a b : Key) : Prop :=
  a.owner≠b.owner ∨ a.address+4≤b.address ∨ b.address+4≤a.address

theorem Key.disjoint_symm {a b : Key} (h : a.disjoint b) : b.disjoint a := by
  rcases h with h | h | h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem Key.disjoint_ne {a b : Key} (h : a.disjoint b) : a≠b := by
  intro same
  subst b
  rcases h with h | h | h <;> simp_all

theorem Key.disjoint_bytes {a b : Key} (h : a.disjoint b) (owner address : Nat)
    (ha : a.covers owner address) : ¬b.covers owner address := by
  rcases ha with ⟨ha,lo,hi⟩
  rintro ⟨hb,lo',hi'⟩
  rcases h with h | h | h
  · exact h (ha.symm.trans hb)
  · omega
  · omega

/-- Ordinary local memory primitives. α is arbitrary (including Real); β is
an observation carrier. Only the FP specialization relates a word to four
finite bytes. There is no encoding of arbitrary Real values into FP32 bytes. -/
structure Ops (M α β : Type) where
  read : M → Key → α
  write : M → Key → α → M
  observe : M → Nat → Nat → β
  read_same : ∀ s key value,read (write s key value) key=value
  read_other : ∀ s writer value observed,observed.disjoint writer →
    read (write s writer value) observed=read s observed
  frame : ∀ s writer value owner address,¬writer.covers owner address →
    observe (write s writer value) owner address=observe s owner address

def stores {M α β ι : Type} (ops : Ops M α β) (key : ι → Key) (values : ι → α) : List ι → M → M
  | [],s => s
  | i::rest,s => stores ops key values rest (ops.write s (key i) (values i))

theorem stores_append {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (first rest : List ι) (s : M) :
    stores ops key values (first++rest) s=stores ops key values rest (stores ops key values first s) := by
  induction first generalizing s with
  | nil => rfl
  | cons i first ih => exact ih _

theorem read_frame {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (observed : Key)
    (outside : ∀ i∈jobs,observed.disjoint (key i)) :
    ops.read (stores ops key values jobs s) observed=ops.read s observed := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun j hj => outside j (by simp [hj])),ops.read_other _ _ _ _ (outside i (by simp))]

theorem stable_read {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (observed : Key) (expected : α)
    (initial : ops.read s observed=expected)
    (consistent : ∀ i∈jobs,key i=observed → values i=expected)
    (separate : ∀ i∈jobs,key i≠observed → observed.disjoint (key i)) :
    ops.read (stores ops key values jobs s) observed=expected := by
  induction jobs generalizing s with
  | nil => exact initial
  | cons i rest ih =>
    apply ih _ _ (fun j hj => consistent j (by simp [hj])) (fun j hj => separate j (by simp [hj]))
    by_cases same : key i=observed
    · rw [same,ops.read_same,consistent i (by simp) same]
    · rw [ops.read_other _ _ _ _ (separate i (by simp) same)]
      exact initial

theorem readback {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (i : ι) (member : i∈jobs)
    (consistent : ∀ j∈jobs,key j=key i → values j=values i)
    (separate : ∀ j∈jobs,key j≠key i → (key i).disjoint (key j)) :
    ops.read (stores ops key values jobs s) (key i)=values i := by
  obtain ⟨before,after,same⟩ := List.append_of_mem member
  rw [same,stores_append,stores]
  apply stable_read ops key values after _ (key i) (values i) (ops.read_same _ _ _)
  · intro j hj
    exact consistent j (by rw [same]; simp [hj])
  · intro j hj
    exact separate j (by rw [same]; simp [hj])

theorem bytes_frame {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (owner address : Nat)
    (outside : ∀ i∈jobs,¬(key i).covers owner address) :
    ops.observe (stores ops key values jobs s) owner address=ops.observe s owner address := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun j hj => outside j (by simp [hj])),ops.frame _ _ _ _ _ (outside i (by simp))]

#axiomsClean readback
#axiomsClean bytes_frame
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Memory

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Raw
open VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Memory
noncomputable section
open Classical

abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev State := Nat → Nat → Byte

def unpack (word : Word) (i : Fin 4) : Byte := word.extractLsb' (8*i.val) 8

def pack (bytes : Fin 4 → Byte) : Word :=
  ((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0

theorem pack_unpack (word : Word) : pack (unpack word)=word := by
  change (((word.extractLsb' 24 8 ++ word.extractLsb' 16 8) ++
    word.extractLsb' 8 8) ++ word.extractLsb' 0 8)=word
  bv_decide

theorem unpack_pack (bytes : Fin 4 → Byte) (i : Fin 4) : unpack (pack bytes) i=bytes i := by
  fin_cases i
  all_goals
    apply BitVec.eq_of_getLsbD_eq
    intro bit
    simp only [unpack,pack,BitVec.getLsbD_extractLsb',BitVec.getLsbD_append]
    by_cases small : bit<8
    · interval_cases bit <;> simp [BitVec.getElem_append,BitVec.getElem_extractLsb']
    · simp [BitVec.getLsbD_of_ge,show 8≤bit by omega]

def read (s : State) (key : Key) : Word :=
  pack (fun i => s key.owner (key.address+i.val))

/-- Four uint8 stores in the source's index order. Arbitrary base alignment
and backing extents are permitted; no whole-backing word view is required. -/
def write (s : State) (key : Key) (word : Word) : State := fun owner address =>
  if owner=key.owner then
    if address=key.address then unpack word 0
    else if address=key.address+1 then unpack word 1
    else if address=key.address+2 then unpack word 2
    else if address=key.address+3 then unpack word 3
    else s owner address
  else s owner address

theorem write_frame (s : State) (key : Key) (word : Word) (owner address : Nat)
    (outside : ¬key.covers owner address) : write s key word owner address=s owner address := by
  unfold write
  split_ifs <;> try rfl
  all_goals exfalso; apply outside; unfold Key.covers; omega

theorem read_same (s : State) (key : Key) (word : Word) : read (write s key word) key=word := by
  have bytes : (fun i : Fin 4 => write s key word key.owner (key.address+i.val))=unpack word := by
    funext i
    fin_cases i <;> simp [write]
  unfold read
  rw [bytes,pack_unpack]

theorem read_other (s : State) (writer : Key) (word : Word) (observed : Key)
    (separate : observed.disjoint writer) : read (write s writer word) observed=read s observed := by
  unfold read
  congr 1
  funext i
  apply write_frame
  exact Key.disjoint_bytes separate observed.owner (observed.address+i.val)
    ⟨rfl,by omega,by omega⟩

def ops : Ops State Word Byte where
  read := read
  write := write
  observe := id
  read_same := read_same
  read_other := read_other
  frame := write_frame

theorem byte_of_read (s : State) (key : Key) (i : Fin 4) :
    s key.owner (key.address+i.val)=unpack (read s key) i :=
  (unpack_pack (fun i => s key.owner (key.address+i.val)) i).symm

#axiomsClean pack_unpack
#axiomsClean unpack_pack
#axiomsClean read_same
#axiomsClean read_other
#axiomsClean write_frame
#axiomsClean byte_of_read
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Raw
