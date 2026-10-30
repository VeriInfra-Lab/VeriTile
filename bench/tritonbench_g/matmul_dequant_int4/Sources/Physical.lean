import bench.tritonbench_g.matmul_dequant_int4.Sources.Memory
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Physical
set_option maxHeartbeats 4000000
abbrev Byte := BitVec 8
inductive Allocation where
  | value (r : Region) | packed (r : PackedRegion)
  deriving DecidableEq
abbrev Bytes := Allocation → Nat → Byte
def encode32 (x : Word) (byte : Fin 4) : Byte := x.extractLsb' (8*byte.val) 8

def decode32 (bytes : Fin 4 → Byte) : Word := bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0

theorem decode32_encode32 (x : Word) : decode32 (encode32 x)=x := by
  change (x.extractLsb' 24 8 ++ x.extractLsb' 16 8 ++ x.extractLsb' 8 8 ++ x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode32_decode32 (bytes : Fin 4 → Byte) : encode32 (decode32 bytes)=bytes := by
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

abbrev Half := BitVec 16
def encode16 (x : Half) (byte : Fin 2) : Byte := x.extractLsb' (8*byte.val) 8
def decode16 (bytes : Fin 2 → Byte) : Half := bytes 1++bytes 0
theorem decode16_encode16 (x : Half) : decode16 (encode16 x)=x := by
  change (x.extractLsb' 8 8++x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self
theorem encode16_decode16 (bytes : Fin 2 → Byte) : encode16 (decode16 bytes)=bytes := by
  funext byte
  fin_cases byte
  · exact BitVec.extractLsb'_append_eq_right
  · exact BitVec.extractLsb'_append_eq_left

/-- Exact proven payload codec, instantiated only by the concrete 16/32-bit
formats below at the public native surface. It adds no numerical assumptions. -/
structure Codec (α : Type) where
  width : Nat
  positive : 0<width
  encode : α → Fin width → Byte
  decode : (Fin width → Byte) → α
  decode_encode : ∀ x,decode (encode x)=x
  encode_decode : ∀ bytes,encode (decode bytes)=bytes
def halfCodec : Codec Half := ⟨2,by decide,encode16,decode16,decode16_encode16,encode16_decode16⟩
def wordCodec : Codec Word := ⟨4,by decide,encode32,decode32,decode32_encode32,encode32_decode32⟩

def load (codec : Codec α) (m : Bytes) : State α :=
  ⟨fun r a => codec.decode (fun i => m (.value r) (codec.width*a+i.val)),
   fun r a => decode32 (fun i => m (.packed r) (4*a+i.val))⟩
def byteStore (codec : Codec α) (m : Bytes) (destination : Region) (address : Nat) (value : α) : Bytes :=
  fun r b => if r=.value destination ∧ b/codec.width=address then
    codec.encode value ⟨b%codec.width,Nat.mod_lt _ codec.positive⟩ else m r b

theorem load_store (codec : Codec α) (m : Bytes) (destination : Region) (address : Nat) (value : α) :
    load codec (byteStore codec m destination address value)=store (load codec m) destination address value := by
  have hd (b : Nat) (i : Fin codec.width) : (codec.width*b+i.val)/codec.width=b := by
    rw [Nat.mul_comm codec.width b]
    exact div_coordinate _ _ _ codec.positive i.isLt
  have hm (b : Nat) (i : Fin codec.width) : (codec.width*b+i.val)%codec.width=i.val := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt i.isLt]
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext r b
    by_cases hr : r=destination
    · subst r
      by_cases ha : b=address
      · subst b
        simp only [load,byteStore,hd,true_and,ite_true]
        have he : (fun i : Fin codec.width => codec.encode value
            ⟨(codec.width*address+i.val)%codec.width,Nat.mod_lt _ codec.positive⟩)=codec.encode value := by
          funext i;congr 1;exact Fin.ext (hm address i)
        rw [he,codec.decode_encode]
      · simp [load,byteStore,hd,ha,store]
    · simp [load,byteStore,hd,hr,store]
  · funext r b;simp [load,byteStore,store]

def step (codec : Codec α) (destination : Region) (columns : Nat)
    (compute : State α → Work → α) (w : Work) (current : Bytes) :=
  byteStore codec current destination (w.row*columns+w.col) (compute (load codec current) w)
def run (codec : Codec α) (destination : Region) (columns : Nat)
    (compute : State α → Work → α) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => run codec destination columns compute ws (step codec destination columns compute w m)

theorem step_load (codec : Codec α) (destination : Region) (columns : Nat)
    (compute : State α → Work → α) (w : Work) (current : Bytes) :
    load codec (step codec destination columns compute w current)=
      Writes.step destination columns compute w (load codec current) := by
  unfold step Writes.step
  rw [load_store]
theorem run_load (codec : Codec α) (destination : Region) (columns : Nat)
    (compute : State α → Work → α) (order : List Work) (current : Bytes) :
    load codec (run codec destination columns compute order current)=
      Writes.run destination columns compute order (load codec current) := by
  induction order generalizing current with
  | nil => rfl
  | cons w ws ih =>
    change load codec (run codec destination columns compute ws (step codec destination columns compute w current))=_
    rw [ih,step_load];rfl
theorem original_byte (codec : Codec α) (m : Bytes) (r : Region) (a : Nat) (i : Fin codec.width) :
    codec.encode ((load codec m).values r a) i=m (.value r) (codec.width*a+i.val) :=
  congrFun (codec.encode_decode (fun j => m (.value r) (codec.width*a+j.val))) i

theorem complete_bytes (codec : Codec α) (destination : Region) (compute : State α → Work → α)
    (stable : Writes.Stable destination compute) (grid : Grid rows columns)
    (order : List Work) (complete : grid.Complete order) (initial : Bytes)
    (w : Work) (hw : w.Active rows columns) (i : Fin codec.width) :
    run codec destination columns compute order initial (.value destination)
      (codec.width*(w.row*columns+w.col)+i.val)=codec.encode (compute (load codec initial) w) i := by
  have h := Writes.complete_result destination compute stable grid order complete (load codec initial) w hw
  rw [←run_load] at h
  have hb := congrArg (fun v => codec.encode v i) h
  exact (original_byte codec (run codec destination columns compute order initial)
    destination (w.row*columns+w.col) i).symm.trans hb

theorem frame (codec : Codec α) (destination : Region) (columns : Nat)
    (compute : State α → Work → α) (order : List Work) (initial : Bytes)
    (r : Allocation) (b : Nat) (outside : ∀ w∈order,¬(r=.value destination ∧ b/codec.width=w.row*columns+w.col)) :
    run codec destination columns compute order initial r b=initial r b := by
  induction order generalizing initial with
  | nil => rfl
  | cons w ws ih =>
    change run codec destination columns compute ws (step codec destination columns compute w initial) r b=_
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    simp [step,byteStore,outside w (by simp)]
theorem complete_frame (codec : Codec α) (destination : Region) (compute : State α → Work → α)
    (grid : Grid rows columns) (order : List Work) (complete : grid.Complete order)
    (initial : Bytes) (r : Allocation) (b : Nat) (outside : r≠.value destination ∨ rows*columns≤b/codec.width) :
    run codec destination columns compute order initial r b=initial r b := by
  apply frame
  intro w hw ⟨hr,ha⟩
  have active := (grid.complete_mem order complete w).mp hw
  have bound := rowMajor_bound rows columns w.row w.col active.1 active.2
  rcases outside with h|h
  · exact h hr
  · omega

#axiomsClean decode32_encode32
#axiomsClean encode32_decode32
#axiomsClean decode16_encode16
#axiomsClean encode16_decode16
#axiomsClean load_store
#axiomsClean run_load
#axiomsClean complete_bytes
#axiomsClean complete_frame
end VeriTile.Bench.Repairs.MatmulDequantInt4.Physical
