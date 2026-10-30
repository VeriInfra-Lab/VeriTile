import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Native

namespace VeriTile.Bench.Repairs.RotaryTransformOps.Physical
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

/-- Mutable payload allocations are distinct regions. Read-only metadata
regions name logical views: runtime cumulative may alias caller CU when
contiguous() returns its argument. These views are never kernel store targets.
Offsets are byte offsets within the represented allocation or view. -/
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

/-- Four physical loads, the selected local graph, then two physical stores.
All operands are loaded before either store. -/
def byteStep (c : Config) (p : Instructions) (w : Work) (m : Bytes) : Bytes :=
  let words := load m
  let even := Native.pairWord c p w words false
  let odd := Native.pairWord c p w words true
  store (store m c.shape.output (c.cell w false) even)
    c.shape.output (c.cell w true) odd

theorem byteStep_load (c : Config) (p : Instructions) (w : Work) (m : Bytes) :
    load (byteStep c p w m)=Native.wordStep c p w (load m) := by
  unfold byteStep Native.wordStep
  rw [load_store,load_store]

def byteExecute (c : Config) (p : Instructions) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => byteExecute c p ws (byteStep c p w m)

theorem byteExecute_load (c : Config) (p : Instructions) (ws : List Work) (m : Bytes) :
    load (byteExecute c p ws m)=Native.wordExecute c p ws (load m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change load (byteExecute c p ws (byteStep c p w m))=_
    rw [ih,byteStep_load]
    rfl

def native (c : Config) (p : Instructions) (m : Bytes) : Bytes := byteExecute c p (schedule c) m

/-- Equality to the interpreted unchanged kernel arithmetic and vector
statement order follows from local graph lowering plus real byte stores. -/
theorem native_source (c : Config) (valid : c.Valid) (p : Instructions) (m : Bytes) :
    load (native c p m)=Native.project (source c (Native.ops p) (Native.liftMemory (load m))) := by
  rw [Native.source_selected c valid]
  exact byteExecute_load c p _ m

theorem native_result (c : Config) (valid : c.Valid) (p : Instructions) (m : Bytes)
    (w : Work) (active : c.Active w) (odd : Bool) :
    load (native c p m) c.shape.output (c.cell w odd)=Native.pairWord c p w (load m) odd := by
  rw [show load (native c p m)=Native.selected c p (load m) from byteExecute_load c p _ m]
  exact Native.selected_readback c valid p (load m) w active odd

theorem native_bytes (c : Config) (valid : c.Valid) (p : Instructions) (m : Bytes)
    (w : Work) (active : c.Active w) (odd : Bool) (i : Fin 4) :
    native c p m c.shape.output (4*c.cell w odd+i.val)=
      encode (Native.pairWord c p w (load m) odd) i := by
  rw [←original_bytes (native c p m) c.shape.output (c.cell w odd) i,
    native_result c valid p m w active odd]

theorem native_frame (c : Config) (p : Instructions) (m : Bytes) (r : Region) (b : Nat)
    (outside : ∀ w,c.Active w → ¬Writes c w r (b/4)) : native c p m r b=m r b := by
  have hf := Native.selected_frame c p (load m) r (b/4) outside
  change Native.wordExecute c p (schedule c) (load m) r (b/4)=load m r (b/4) at hf
  rw [←byteExecute_load c p (schedule c) m] at hf
  have hb := congrArg (fun value => encode value ⟨b%4,Nat.mod_lt _ (by decide)⟩) hf
  dsimp only at hb
  rw [original_bytes,original_bytes] at hb
  have eq : 4*(b/4)+b%4=b := by omega
  simpa [eq,native] using hb

theorem native_tail (c : Config) (valid : c.Valid) (p : Instructions) (m : Bytes)
    (n h d : Nat) (hh : h<c.shape.heads) (hd : d<c.shape.width) (tail : 2*c.shape.half≤d)
    (i : Fin 4) : native c p m c.shape.output (4*c.shape.address n h d+i.val)=
      m c.shape.output (4*c.shape.address n h d+i.val) := by
  apply native_frame
  have eq : (4*c.shape.address n h d+i.val)/4=c.shape.address n h d := by omega
  rw [eq]
  exact tail_outside c valid n h d hh hd tail

theorem native_readonly (c : Config) (p : Instructions) (m : Bytes) (r : Region) (b : Nat)
    (hr : r≠.input ∧ r≠.output) : native c p m r b=m r b := by
  apply native_frame
  intro w _
  cases hi : c.shape.inplace <;> simp_all [Writes,Shape.output]

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean load_store
#axiomsClean native_source
#axiomsClean native_result
#axiomsClean native_bytes
#axiomsClean native_frame
#axiomsClean native_tail
#axiomsClean native_readonly
end VeriTile.Bench.Repairs.RotaryTransformOps.Physical
