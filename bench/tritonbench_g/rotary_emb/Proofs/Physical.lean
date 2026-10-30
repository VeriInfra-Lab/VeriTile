import bench.tritonbench_g.rotary_emb.Proofs.Native

namespace VeriTile.Bench.Repairs.RotaryEmb.Physical
open Native
abbrev Byte := BitVec 8
abbrev Bytes := Memory Byte

def encode (x : Word) (byte : Fin 4) : Byte := x.extractLsb' (8*byte.val) 8

def decode (bytes : Fin 4 → Byte) : Word := bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0

theorem decode_encode (x : Word) : decode (encode x)=x := by
  change (x.extractLsb' 24 8 ++ x.extractLsb' 16 8 ++ x.extractLsb' 8 8 ++ x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4 → Byte) : encode (decode bytes)=bytes := by
  funext byte
  fin_cases byte
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

/-- Allocation identities are distinct regions. Offset is a byte offset in
that allocation, not a fabricated tensor coordinate or an output oracle. -/
def load (m : Bytes) : Memory Word := fun r a => decode (fun i => m r (4*a+i.val))

/-- Exact little-endian aligned four-byte store. -/
def store (m : Bytes) (r : Region) (a : Nat) (value : Word) : Bytes :=
  fun q b => if q=r ∧ b/4=a then encode value ⟨b%4,Nat.mod_lt _ (by decide)⟩ else m q b

theorem load_store (m : Bytes) (r : Region) (a : Nat) (value : Word) :
    load (store m r a value)=Stores.store (load m) ((r,a),value) := by
  funext q b
  have div (i : Fin 4) : (4*b+i.val)/4=b := by omega
  have rem (i : Fin 4) : (4*b+i.val)%4=i.val := by omega
  by_cases hr : q=r
  · subst q
    by_cases ha : b=a
    · subst b
      simp only [load,store,div,true_and,ite_true]
      have eq : (fun i : Fin 4 => encode value ⟨(4*a+i.val)%4,by omega⟩)=encode value := by
        funext i
        congr 1
        exact Fin.ext (rem i)
      rw [eq,decode_encode]
      simp [Stores.store]
    · simp [load,store,div,ha,Stores.store]
  · simp [load,store,hr,Stores.store]

theorem original_bytes (m : Bytes) (r : Region) (a : Nat) (i : Fin 4) :
    encode (load m r a) i=m r (4*a+i.val) :=
  congrFun (encode_decode (fun i => m r (4*a+i.val))) i

theorem store_frame (m : Bytes) (r : Region) (a : Nat) (v : Word) (q : Region) (b : Nat)
    (outside : ¬(q=r ∧ b/4=a)) : store m r a v q b=m q b := by
  simp [store,outside]

/-- Six physical loads, the selected local graph, then two physical stores.
All operands are loaded before either store. -/
def byteStep (s : Shape) (p : Instructions) (w : Work) (m : Bytes) : Bytes :=
  let words := load m
  let even := Native.pairWord s p w words false
  let odd := Native.pairWord s p w words true
  store (store m (dataRegion w.key) (s.cell w false) even)
    (dataRegion w.key) (s.cell w true) odd

theorem byteStep_load (s : Shape) (p : Instructions) (w : Work) (m : Bytes) :
    load (byteStep s p w m)=Native.wordStep s p w (load m) := by
  unfold byteStep Native.wordStep
  rw [load_store,load_store]

def byteExecute (s : Shape) (p : Instructions) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => byteExecute s p ws (byteStep s p w m)

theorem byteExecute_load (s : Shape) (p : Instructions) (ws : List Work) (m : Bytes) :
    load (byteExecute s p ws m)=Native.wordExecute s p ws (load m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change load (byteExecute s p ws (byteStep s p w m))=_
    rw [ih,byteStep_load]
    rfl

def native (s : Shape) (p : Instructions) (m : Bytes) : Bytes := byteExecute s p (schedule s) m

/-- Equality to the interpreted unmodified source arithmetic and vector
statement order follows from local graph lowering plus real byte stores. -/
theorem native_source (s : Shape) (valid : s.Valid) (p : Instructions) (m : Bytes) :
    load (native s p m)=Native.project (source s (Native.ops p) (Native.liftMemory (load m))) := by
  rw [Native.source_selected s valid]
  exact byteExecute_load s p _ m

theorem native_result (s : Shape) (valid : s.Valid) (p : Instructions) (m : Bytes)
    (w : Work) (active : s.Active w) (odd : Bool) :
    load (native s p m) (dataRegion w.key) (s.cell w odd)=Native.pairWord s p w (load m) odd := by
  rw [show load (native s p m)=Native.selected s p (load m) from byteExecute_load s p _ m]
  exact Native.selected_readback s valid p (load m) w active odd

theorem native_bytes (s : Shape) (valid : s.Valid) (p : Instructions) (m : Bytes)
    (w : Work) (active : s.Active w) (odd : Bool) (i : Fin 4) :
    native s p m (dataRegion w.key) (4*s.cell w odd+i.val)=
      encode (Native.pairWord s p w (load m) odd) i := by
  rw [←original_bytes (native s p m) (dataRegion w.key) (s.cell w odd) i,
    native_result s valid p m w active odd]

theorem native_frame (s : Shape) (p : Instructions) (m : Bytes) (r : Region) (b : Nat)
    (outside : ∀ w,s.Active w → ¬Writes s w r (b/4)) : native s p m r b=m r b := by
  have hf := Native.selected_frame s p (load m) r (b/4) outside
  change Native.wordExecute s p (schedule s) (load m) r (b/4)=load m r (b/4) at hf
  rw [←byteExecute_load s p (schedule s) m] at hf
  have hb := congrArg (fun value => encode value ⟨b%4,Nat.mod_lt _ (by decide)⟩) hf
  dsimp only at hb
  rw [original_bytes,original_bytes] at hb
  have eq : 4*(b/4)+b%4=b := by omega
  simpa [eq,native] using hb

theorem native_tail (s : Shape) (valid : s.Valid) (p : Instructions) (m : Bytes)
    (key : Bool) (n h d : Nat) (hh : h<s.heads key) (hd : d<s.width) (tail : s.rotary≤d)
    (i : Fin 4) : native s p m (dataRegion key) (4*s.address key n h d+i.val)=
      m (dataRegion key) (4*s.address key n h d+i.val) := by
  apply native_frame
  have eq : (4*s.address key n h d+i.val)/4=s.address key n h d := by omega
  rw [eq]
  exact tail_outside s valid key n h d hh hd tail

theorem native_coefficients (s : Shape) (p : Instructions) (m : Bytes) (sine : Bool) (b : Nat) :
    native s p m (if sine then .sin else .cos) b=m (if sine then .sin else .cos) b := by
  apply native_frame
  intro w _
  cases sine <;> cases hk : w.key <;> simp [Writes,dataRegion,hk]

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean load_store
#axiomsClean native_source
#axiomsClean native_result
#axiomsClean native_bytes
#axiomsClean native_frame
#axiomsClean native_tail
#axiomsClean native_coefficients
end VeriTile.Bench.Repairs.RotaryEmb.Physical
