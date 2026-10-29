import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev Memory := Nat → Nat → Byte

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

def read (s : Memory) (key : Key) : Word :=
  pack (fun i => s key.owner (key.address+i.val))

/-- Four uint8 stores in the source's index order. Arbitrary base alignment
and backing extents are permitted; no whole-backing word view is required. -/
def write (s : Memory) (key : Key) (word : Word) : Memory := fun owner address =>
  if owner=key.owner then
    if address=key.address then unpack word 0
    else if address=key.address+1 then unpack word 1
    else if address=key.address+2 then unpack word 2
    else if address=key.address+3 then unpack word 3
    else s owner address
  else s owner address

theorem write_frame (s : Memory) (key : Key) (word : Word) (owner address : Nat)
    (outside : ¬key.covers owner address) : write s key word owner address=s owner address := by
  unfold write
  split_ifs <;> try rfl
  all_goals exfalso; apply outside; unfold Key.covers; omega

theorem read_same (s : Memory) (key : Key) (word : Word) : read (write s key word) key=word := by
  have bytes : (fun i : Fin 4 => write s key word key.owner (key.address+i.val))=unpack word := by
    funext i
    fin_cases i <;> simp [write]
  unfold read
  rw [bytes,pack_unpack]

theorem read_other (s : Memory) (writer : Key) (word : Word) (observed : Key)
    (separate : observed.disjoint writer) : read (write s writer word) observed=read s observed := by
  unfold read
  congr 1
  funext i
  apply write_frame
  exact Key.disjoint_bytes separate observed.owner (observed.address+i.val)
    ⟨rfl,by omega,by omega⟩

def ops : Ops Memory Word Byte where
  read := read
  write := write
  observe := id
  read_same := read_same
  read_other := read_other
  frame := write_frame

/-- Lazy negation is exact sign-bit XOR, including zeros and all NaN payloads.
It is independent of native floating negation and subtraction instructions. -/
def sign : Negation Word where
  apply := fun word => word ^^^ BitVec.ofNat 32 0x80000000
  involutive := by intro word; simp [BitVec.xor_assoc]

def xorHigh (bytes : Fin 4 → Byte) : Fin 4 → Byte :=
  fun i => if i.val=3 then bytes i ^^^ BitVec.ofNat 8 0x80 else bytes i

theorem xor_transport (word : Word) : pack (xorHigh (unpack word))=sign.apply word := by
  change (((((word.extractLsb' 24 8) ^^^ BitVec.ofNat 8 0x80) ++
    word.extractLsb' 16 8) ++ word.extractLsb' 8 8) ++ word.extractLsb' 0 8)=
    word ^^^ BitVec.ofNat 32 0x80000000
  apply BitVec.eq_of_getLsbD_eq
  intro bit
  simp only [BitVec.getLsbD_append,BitVec.getLsbD_xor,BitVec.getLsbD_extractLsb']
  by_cases small : bit<32
  · interval_cases bit <;> simp
  · simp [BitVec.getLsbD_of_ge,show 32≤bit by omega,
      show ¬bit<8 by omega,show ¬bit<16 by omega,show ¬bit<24 by omega,
      show ¬bit-24<8 by omega]

/-- The uint8 index_select, followed by optional [3::4] XOR and a fresh byte
copy, yields exactly the logical word. Copy identity is an ordinary local
byte-transfer contract, never a callback supplying the optimizer output. -/
def gather (s : Memory) (view : View) (i : Nat) : Word :=
  let bytes := fun j : Fin 4 => s (view.key i).owner ((view.key i).address+j.val)
  pack (if view.negative then xorHigh bytes else bytes)

theorem gather_logical (s : Memory) (view : View) (i : Nat) :
    gather s view i=logical ops sign s view i := by
  cases flag : view.negative
  · simp [gather,logical,ops,Negation.encode,flag,read]
  · simp only [gather,flag,ite_true,logical,Negation.encode,ops]
    have bytes : (fun j : Fin 4 => s (view.key i).owner ((view.key i).address+j.val))=
        unpack (read s (view.key i)) := by
      funext j
      exact (unpack_pack (fun j => s (view.key i).owner ((view.key i).address+j.val)) j).symm
    rw [bytes,xor_transport]

theorem byte_of_read (s : Memory) (key : Key) (i : Fin 4) :
    s key.owner (key.address+i.val)=unpack (read s key) i :=
  (unpack_pack (fun i => s key.owner (key.address+i.val)) i).symm

#axiomsClean pack_unpack
#axiomsClean unpack_pack
#axiomsClean read_same
#axiomsClean read_other
#axiomsClean xor_transport
#axiomsClean gather_logical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes
