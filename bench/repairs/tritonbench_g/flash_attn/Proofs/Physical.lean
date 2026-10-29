import bench.repairs.tritonbench_g.flash_attn.Proofs.Native
namespace VeriTile.Bench.Repairs.FlashAttn.Physical
set_option maxHeartbeats 4000000
abbrev Byte := BitVec 8
abbrev Bytes := Region → Nat → Byte
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


def load (m : Bytes) : Memory Word := fun r a=>decode (fun i=>m r (4*a+i.val))
def byteStore (m : Bytes) (target : Region × Nat) (v : Word) : Bytes :=
  fun r b=>if (r,b/4)=target then encode v ⟨b%4,Nat.mod_lt _ (by decide)⟩ else m r b

theorem load_store (m : Bytes) (target : Region × Nat) (v : Word) :
    load (byteStore m target v)=store (load m) target v := by
  funext r a
  have hd (i : Fin 4) : (4*a+i.val)/4=a := by omega
  have hm (i : Fin 4) : (4*a+i.val)%4=i.val := by omega
  by_cases he : (r,a)=target
  · simp only [load,byteStore,hd,he,ite_true,store]
    have eq : (fun i : Fin 4=>encode v ⟨(4*a+i.val)%4,by omega⟩)=encode v := by
      funext i;congr 1;exact Fin.ext (hm i)
    rw [eq,decode_encode]
  · simp [load,byteStore,hd,he,store]

def byteStep (x : Layout) (p : Ops Word) (scale : Word) (w : Work) (m : Bytes) :=
  byteStore m (w.target x) (w.result x p (load m) scale)
def byteExecute (x : Layout) (p : Ops Word) (scale : Word) : List Work → Bytes → Bytes
  | [],m=>m
  | w::ws,m=>byteExecute x p scale ws (byteStep x p scale w m)

theorem execute_load (x : Layout) (p : Ops Word) (scale : Word) (ws : List Work) (m : Bytes) :
    load (byteExecute x p scale ws m)=execute x p scale ws (load m) := by
  induction ws generalizing m with
  | nil=>rfl
  | cons w ws ih=>
    change load (byteExecute x p scale ws (byteStep x p scale w m))=_
    rw [ih]
    simp only [byteStep,load_store,execute,step]

theorem original_byte (m : Bytes) (r : Region) (a : Nat) (i : Fin 4) :
    encode (load m r a) i=m r (4*a+i.val) :=
  congrFun (encode_decode (fun j=>m r (4*a+j.val))) i

theorem whole_bytes (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x)
    (actual reference : Native.Instructions) (numeric : Native.Correspondence actual reference)
    (scale : Word) (ws : List Work) (complete : Complete x.shape ws) (m : Bytes)
    (w : Work) (active : w.active x.shape) (i : Fin 4) :
    byteExecute x (Native.ops actual) scale ws m (w.target x).1 (4*(w.target x).2+i.val)=
      encode (w.result x (Native.ops reference) (load m) scale) i := by
  have h:=Native.whole_native x hs ho actual reference numeric (load m) scale ws complete w active
  rw [←execute_load] at h
  have hb:=congrArg (fun v=>encode v i) h
  exact (original_byte _ _ _ i).symm.trans hb

theorem execute_frame_bytes (x : Layout) (p : Ops Word) (scale : Word) (ws : List Work)
    (m : Bytes) (r : Region) (b : Nat) (outside : ∀ w∈ws,(r,b/4)≠w.target x) :
    byteExecute x p scale ws m r b=m r b := by
  induction ws generalizing m with
  | nil=>rfl
  | cons w ws ih=>
    change byteExecute x p scale ws (byteStep x p scale w m) r b=_
    rw [ih _ (fun v hv=>outside v (by simp [hv]))]
    simp [byteStep,byteStore,outside w (by simp)]

theorem whole_frame_bytes (x : Layout) (hs : x.shape.Valid) (p : Ops Word) (scale : Word)
    (ws : List Work) (complete : Complete x.shape ws) (m : Bytes) (r : Region) (b : Nat)
    (outside : ¬footprint x r (b/4)) : byteExecute x p scale ws m r b=m r b := by
  apply execute_frame_bytes
  intro w hw he
  exact outside ⟨w,(schedule_mem x.shape hs w).mp (complete.mem_iff.mp hw),he⟩
#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean load_store
#axiomsClean execute_load
#axiomsClean whole_bytes
#axiomsClean whole_frame_bytes
end VeriTile.Bench.Repairs.FlashAttn.Physical
