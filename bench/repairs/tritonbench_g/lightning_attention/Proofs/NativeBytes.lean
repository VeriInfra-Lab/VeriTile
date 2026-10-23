import bench.repairs.tritonbench_g.lightning_attention.Proofs.SourceHost
namespace VeriTile.Bench.Repairs.LightningAttention.NativeBytes
open VeriTile Triton FP.Structural Host Native Checked
open VeriTile.Bench.Repairs.ChunkLinearAttn
open Classical
noncomputable section
abbrev Byte := BitVec 8
/-- Byte-addressed backing memory; live addresses are bounded by PointerBinding
and the checked signed64 pointer room, proved below. -/
abbrev Bytes := Nat → Byte

def encode (x : Word) (i : Fin 4) : Byte := x.extractLsb' (8*i.val) 8
def decode (bytes : Fin 4 → Byte) : Word := bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0

theorem decode_encode (x : Word) : decode (encode x)=x := by
  change (x.extractLsb' 24 8 ++ x.extractLsb' 16 8 ++ x.extractLsb' 8 8 ++ x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4 → Byte) : encode (decode bytes)=bytes := by
  funext i
  fin_cases i
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 0 8=bytes 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

def load (m : Bytes) (wordAddress : Nat) : Word := decode (fun i=>m (4*wordAddress+i.val))
def fromBytes (m : Bytes) : Nat → Cell Word := fun a=>.mk .real (load m a)

theorem original_bytes (m : Bytes) (a : Nat) (i : Fin 4) : encode (load m a) i=m (4*a+i.val) :=
  congrFun (encode_decode (fun i=>m (4*a+i.val))) i

/-- One little-endian aligned FP32 store. Exactly four backing bytes change. -/
def store (m : Bytes) (wordAddress : Nat) (v : Word) : Bytes :=
  fun a=>if a/4=wordAddress then encode v ⟨a%4,Nat.mod_lt _ (by decide)⟩ else m a

theorem load_store (m : Bytes) (a b : Nat) (v : Word) :
    load (store m a v) b=if b=a then v else load m b := by
  have div (i : Fin 4) : (4*b+i.val)/4=b := by omega
  have rem (i : Fin 4) : (4*b+i.val)%4=i.val := by omega
  unfold load store
  simp only [div]
  split_ifs with h
  · have eq : (fun i : Fin 4=>encode v ⟨(4*b+i.val)%4,Nat.mod_lt _ (by decide)⟩)=encode v := by
      funext i;congr 1;exact Fin.ext (rem i)
    rw [eq,decode_encode]
  · rfl

theorem store_frame (m : Bytes) (a byte : Nat) (v : Word) (outside : byte/4≠a) :
    store m a v byte=m byte := by simp [store,outside]

def Contains (b : Physical.Buffer) (byte : Nat) : Prop := 4*b.base≤byte ∧ byte<4*(b.base+b.words)

theorem contains_iff (b : Physical.Buffer) (byte : Nat) : Contains b byte ↔ Physical.Contains b (byte/4) := by
  simp only [Contains,Physical.Contains]
  omega

/-- The byte form of a buffer's sequence of four-byte stores. Every byte
selects exactly its owning store and its little-endian component. -/
def write (b : Physical.Buffer) (values : WordMemory) (m : Bytes) : Bytes :=
  fun byte=>if Physical.Contains b (byte/4) then
    encode (values b.region (byte/4-b.base)) ⟨byte%4,Nat.mod_lt _ (by decide)⟩ else m byte

def publish (buffers : List Physical.Buffer) (values : WordMemory) (m : Bytes) : Bytes :=
  buffers.foldl (fun bytes b=>write b values bytes) m

theorem write_succ (b : Physical.Buffer) (n : Nat) (values : WordMemory) (m : Bytes) :
    write {b with words:=n+1} values m=
      store (write {b with words:=n} values m) (b.base+n) (values b.region n) := by
  funext byte
  by_cases eq : byte/4=b.base+n
  · simp [write,store,Physical.Contains,eq]
  · have same : (b.base≤byte/4 ∧ byte/4<b.base+(n+1)) ↔
        (b.base≤byte/4 ∧ byte/4<b.base+n) := by omega
    simp only [write,store,eq,ite_false,Physical.Contains,same]

theorem write_stores (b : Physical.Buffer) (n : Nat) (values : WordMemory) (m : Bytes) :
    write {b with words:=n} values m=
      (List.range n).foldl (fun bytes off=>store bytes (b.base+off) (values b.region off)) m := by
  induction n with
  | zero =>
    funext byte
    simp [write,Physical.Contains]
  | succ n ih =>
    rw [write_succ,ih,List.range_succ,List.foldl_append]
    rfl

theorem load_write (b : Physical.Buffer) (values : WordMemory) (m : Bytes) :
    fromBytes (write b values m)=Physical.write b (Checked.cells values) (fromBytes m) := by
  funext a
  have div (i : Fin 4) : (4*a+i.val)/4=a := by omega
  have rem (i : Fin 4) : (4*a+i.val)%4=i.val := by omega
  by_cases h : Physical.Contains b a
  · have eq : (fun i : Fin 4=>encode (values b.region (a-b.base)) ⟨(4*a+i.val)%4,Nat.mod_lt _ (by decide)⟩)=
        encode (values b.region (a-b.base)) := by
      funext i;congr 1;exact Fin.ext (rem i)
    simp only [fromBytes,load,write,div,h,ite_true,eq,decode_encode]
    simp only [Physical.write,Physical.Contains] at h ⊢
    simp [h,Checked.cells]
  · simp only [fromBytes,load,write,div,h,ite_false]
    simp only [Physical.write,Physical.Contains] at h ⊢
    simp only [h,ite_false,fromBytes,load]

theorem load_publish (buffers : List Physical.Buffer) (values : WordMemory) (m : Bytes) :
    fromBytes (publish buffers values m)=Physical.publish buffers (Checked.cells values) (fromBytes m) := by
  induction buffers generalizing m with
  | nil => rfl
  | cons b bs ih =>
    change fromBytes (publish bs values (write b values m))=Physical.publish bs (Checked.cells values)
      (Physical.write b (Checked.cells values) (fromBytes m))
    rw [ih,load_write]

theorem publish_frame (buffers : List Physical.Buffer) (values : WordMemory) (m : Bytes)
    (byte : Nat) (outside : ∀ b∈buffers,¬Contains b byte) : publish buffers values m byte=m byte := by
  induction buffers generalizing m with
  | nil => rfl
  | cons b bs ih =>
    change publish bs values (write b values m) byte=m byte
    rw [ih _ (fun x hx=>outside x (by simp [hx]))]
    have h := outside b (by simp)
    rw [contains_iff] at h
    simp [write,h]

/-- Bounded byte address of any source-formed word. PointerBinding relates the
actual Python data_ptr byte value to the retained word-indexed host descriptor. -/
theorem live_byte_bound (d : Dimensions) (o : Observations) (ts : List Tensor)
    (admitted : InputContract d o ts) (binding : PointerBinding o ts)
    (t : Tensor) (ht : t∈ts) (off : Nat) (hoff : off<d.formedLimit) (i : Fin 4) :
    4*(t.base+off)+i.val=o.pointer t+4*off+i.val ∧ 4*(t.base+off)+i.val<2^63 := by
  have room := pointer_room d o ts admitted binding t ht
  have base := binding t ht
  constructor
  · omega
  · norm_num at room ⊢;omega

/-- Active input words are decoded from the original four backing bytes.
No independently supplied word memory is accepted by the byte endpoints. -/
theorem input_word (a : Allocations) (m : Bytes) (fallback : Memory Word)
    (r : RegionName) (t : Tensor) (descriptor : Host.descriptor a r=some t) (off : Nat) :
    logicalWords a (fromBytes m) fallback r off=decode (fun i=>m (4*(t.base+off)+i.val)) := by
  simp [logicalWords,Host.logical,descriptor,fromBytes,load]

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean load_store
#axiomsClean write_stores
#axiomsClean load_publish
#axiomsClean publish_frame
#axiomsClean live_byte_bound
#axiomsClean input_word
end
end VeriTile.Bench.Repairs.LightningAttention.NativeBytes
