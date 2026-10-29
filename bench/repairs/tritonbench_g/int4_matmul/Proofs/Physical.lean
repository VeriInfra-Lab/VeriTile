import bench.repairs.tritonbench_g.int4_matmul.Proofs.Native
namespace VeriTile.Bench.Repairs.Int4Matmul.Physical
set_option maxHeartbeats 4000000
abbrev Byte := BitVec 8
inductive Allocation where
  | value (r : Region) | packed (r : PackedRegion)
  deriving DecidableEq
abbrev Bytes := Allocation → Nat → Byte
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


def load (m : Bytes) : State Word :=
  ⟨fun r a => decode (fun i => m (.value r) (4*a+i.val)),
   fun r a => decode (fun i => m (.packed r) (4*a+i.val))⟩
def byteStore (m : Bytes) (a : Nat) (v : Word) : Bytes :=
  fun r b => if r=.value .output ∧ b/4=a then encode v ⟨b%4,Nat.mod_lt _ (by decide)⟩ else m r b

theorem load_store (m : Bytes) (a : Nat) (v : Word) :
    load (byteStore m a v)=store (load m) .output a v := by
  have hd (b : Nat) (i : Fin 4) : (4*b+i.val)/4=b := by omega
  have hm (b : Nat) (i : Fin 4) : (4*b+i.val)%4=i.val := by omega
  change State.mk _ _ = State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext r b
    by_cases hr : r=.output
    · subst r
      by_cases ha : b=a
      · subst b
        simp only [load,byteStore,hd,true_and,ite_true]
        have eq : (fun i : Fin 4 => encode v ⟨(4*a+i.val)%4,by omega⟩)=encode v := by
          funext i;congr 1;exact Fin.ext (hm a i)
        rw [eq,decode_encode]
      · simp [load,byteStore,hd,ha,store]
    · simp [load,byteStore,hr,store]
  · funext r b;simp [load,byteStore,store]

def byteStep (s : Shape) (t : Tile) (p : Ops Word) (w : Work) (m : Bytes) : Bytes :=
  let words := load m
  let v := partialValue s t p words w
  let a := cAddress s w.row w.col
  byteStore m a (if t.splits=1 then v else p.atomicAdd (words.values .output a) v)

theorem step_load (s : Shape) (t : Tile) (p : Ops Word) (w : Work) (m : Bytes) :
    load (byteStep s t p w m)=step s t p w (load m) := by
  unfold byteStep step
  rw [load_store]

def byteExecute (s : Shape) (t : Tile) (p : Ops Word) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => byteExecute s t p ws (byteStep s t p w m)

theorem execute_load (s : Shape) (t : Tile) (p : Ops Word) (ws : List Work) (m : Bytes) :
    load (byteExecute s t p ws m)=execute s t p ws (load m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change load (byteExecute s t p ws (byteStep s t p w m))=_
    rw [ih,step_load];rfl

def byteZero (s : Shape) (m : Bytes) : Bytes :=
  fun r b => if r=.value .output ∧ b/4<s.m*s.n then 0 else m r b

theorem zero_load (s : Shape) (p : Native.Instructions) (isa : Native.ISA) (m : Bytes) :
    load (byteZero s m)=zeroOutput s (Native.ops p isa) (load m) := by
  change State.mk _ _ = State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext r a
    have hd (i : Fin 4) : (4*a+i.val)/4=a := by omega
    by_cases hr : r=.output
    · subst r
      by_cases ha : a<s.m*s.n
      · simp only [byteZero,hd,ha,true_and,ite_true]
        change decode (fun _ => 0)=0
        rfl
      · simp [load,byteZero,hd,ha,zeroOutput]
    · simp [load,byteZero,hd,hr,zeroOutput]
  · funext r a;simp [load,byteZero,zeroOutput]

def byteCall (s : Shape) (t : Tile) (p : Native.Instructions) (isa : Native.ISA)
    (ws : List Work) (m : Bytes) := byteExecute s t (Native.ops p isa) ws (byteZero s m)

theorem call_load (s : Shape) (t : Tile) (p : Native.Instructions) (isa : Native.ISA)
    (ws : List Work) (m : Bytes) :
    load (byteCall s t p isa ws m)=call s t (Native.ops p isa) ws (load m) := by
  unfold byteCall call
  rw [execute_load,zero_load]

theorem original_byte (m : Bytes) (r : Allocation) (a : Nat) (i : Fin 4) :
    encode (decode (fun j => m r (4*a+j.val))) i=m r (4*a+i.val) :=
  congrFun (encode_decode (fun j => m r (4*a+j.val))) i

theorem call_bytes (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (p : Native.Instructions) (isa : Native.ISA) (ws : List Work) (complete : Complete s t ws)
    (m : Bytes) (r c : Nat) (hr : r<s.m) (hc : c<s.n) (i : Fin 4) :
    byteCall s t p isa ws m (.value .output) (4*cAddress s r c+i.val)=
      encode (referenceOutput s t (Native.ops p isa) (load m) r c ws 0) i := by
  have h := call_result s t hs ht (Native.ops p isa) ws complete (load m) r c hr hc
  rw [←call_load] at h
  have hb := congrArg (fun v => encode v i) h
  exact (original_byte (byteCall s t p isa ws m) (.value .output) (cAddress s r c) i).symm.trans hb

theorem execute_frame_bytes (s : Shape) (t : Tile) (p : Ops Word) (ws : List Work)
    (m : Bytes) (r : Allocation) (b : Nat)
    (outside : ∀ w∈ws,¬(r=.value .output ∧ b/4=cAddress s w.row w.col)) :
    byteExecute s t p ws m r b=m r b := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change byteExecute s t p ws (byteStep s t p w m) r b=_
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    simp [byteStep,byteStore,outside w (by simp)]

theorem call_frame_bytes (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (p : Native.Instructions) (isa : Native.ISA) (ws : List Work) (complete : Complete s t ws)
    (m : Bytes) (r : Allocation) (b : Nat) (outside : r≠.value .output ∨ s.m*s.n≤b/4) :
    byteCall s t p isa ws m r b=m r b := by
  unfold byteCall
  rw [execute_frame_bytes]
  · rcases outside with h|h
    · simp [byteZero,h]
    · simp [byteZero,not_lt_of_ge h]
  · intro w hw ⟨hr,ha⟩
    have active := (complete_active s t hs ht ws complete w).mp hw
    have bound := rowMajor_bound s.m s.n w.row w.col active.1 active.2.1
    rcases outside with h|h
    · exact h hr
    · unfold cAddress at ha;omega
#axiomsClean call_frame_bytes
#axiomsClean load_store
#axiomsClean execute_load
#axiomsClean zero_load
#axiomsClean call_load
#axiomsClean call_bytes
end VeriTile.Bench.Repairs.Int4Matmul.Physical
