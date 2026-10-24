import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Native

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Bytes

abbrev Byte := Fin 256
abbrev Word := BitVec 32
abbrev Heap := Nat → Byte

/-- Actual little-endian four-byte encoding, retaining every FP32 bit. -/
def encode (word : Word) (byte : Fin 4) : Byte :=
  ⟨(word.toNat / (256 ^ byte.val)) % 256, Nat.mod_lt _ (by decide)⟩

def decode (bytes : Fin 4 → Byte) : Word :=
  BitVec.ofNat 32 ((bytes 0).val + 256 * (bytes 1).val +
    65536 * (bytes 2).val + 16777216 * (bytes 3).val)

theorem decode_encode (word : Word) : decode (encode word) = word := by
  apply BitVec.eq_of_toNat_eq
  have h := word.isLt
  change (_ % 4294967296) = word.toNat
  simp only [decode, encode, BitVec.toNat_ofNat, Fin.val_zero, Fin.val_one,
    Fin.reduceFinMk, pow_zero, pow_one, Nat.div_one]
  norm_num at h ⊢
  omega

def storeWord (heap : Heap) (base : Nat) (word : Word) : Heap := fun address =>
  if h : base ≤ address ∧ address < base + 4 then
    encode word ⟨address - base, by omega⟩
  else heap address

def loadWord (heap : Heap) (base : Nat) : Word := decode (fun byte => heap (base + byte.val))

theorem stored_byte (heap : Heap) (base : Nat) (word : Word) (byte : Fin 4) :
    storeWord heap base word (base + byte.val) = encode word byte := by
  unfold storeWord
  have h : base ≤ base + byte.val ∧ base + byte.val < base + 4 := by omega
  rw [dif_pos h]
  congr 1
  apply Fin.ext
  change base + byte.val - base = byte.val
  omega

theorem store_readback (heap : Heap) (base : Nat) (word : Word) :
    loadWord (storeWord heap base word) base = word := by
  unfold loadWord
  simp_rw [stored_byte]
  exact decode_encode word

theorem store_frame (heap : Heap) (base : Nat) (word : Word) (address : Nat)
    (h : address < base ∨ base + 4 ≤ address) :
    storeWord heap base word address = heap address := by
  unfold storeWord
  rw [dif_neg (by omega)]

def storeMany (heap : Heap) (writes : List (Nat × Word)) : Heap :=
  writes.foldl (fun current write => storeWord current write.1 write.2) heap

/-- All bytes outside every written four-byte interval are preserved. This
includes allocation padding and trailing bytes, not just tensor element cells. -/
theorem storeMany_frame (heap : Heap) (writes : List (Nat × Word)) (address : Nat)
    (h : ∀ write ∈ writes, address < write.1 ∨ write.1 + 4 ≤ address) :
    storeMany heap writes address = heap address := by
  induction writes generalizing heap with
  | nil => rfl
  | cons write tail ih =>
    change storeMany (storeWord heap write.1 write.2) tail address = heap address
    rw [ih _ (by intro item hi; exact h item (List.mem_cons_of_mem _ hi))]
    exact store_frame heap write.1 write.2 address (h write (by simp))

structure Span where
  base : Nat
  bytes : Nat
  native : base + bytes ≤ 2 ^ 64

def Span.contains (span : Span) (address : Nat) : Prop :=
  span.base ≤ address ∧ address < span.base + span.bytes

def Span.disjoint (left right : Span) : Prop :=
  left.bytes = 0 ∨ right.bytes = 0 ∨
    left.base + left.bytes ≤ right.base ∨ right.base + right.bytes ≤ left.base

def Frame (live : List Span) (before after : Heap) : Prop :=
  ∀ span ∈ live, ∀ address, span.contains address → after address = before address

theorem fresh_output_frame (heap : Heap) (live : List Span) (output : Span)
    (fresh : ∀ span ∈ live, span.disjoint output)
    (writes : List (Nat × Word))
    (bounded : ∀ write ∈ writes, output.base ≤ write.1 ∧ write.1 + 4 ≤ output.base + output.bytes) :
    Frame live heap (storeMany heap writes) := by
  intro span hspan address ha
  apply storeMany_frame
  intro write hw
  have hb := bounded write hw
  have hd := fresh span hspan
  unfold Span.contains at ha
  unfold Span.disjoint at hd
  omega

/-- A nonempty successful write changes an actual byte and has exact full-word
readback, while adjacent bytes remain untouched. This is not a fail-all model. -/
theorem concrete_write_witness :
    let heap : Heap := fun _ => 0
    let after := storeWord heap 4096 (BitVec.ofNat 32 1065353216)
    loadWord after 4096 = BitVec.ofNat 32 1065353216 ∧
      after 4099 = 63 ∧ after 4099 ≠ heap 4099 ∧
      after 4095 = heap 4095 ∧ after 4100 = heap 4100 := by
  refine ⟨store_readback _ _ _, ?_, ?_, ?_, ?_⟩ <;> decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Bytes
