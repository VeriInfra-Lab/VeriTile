import bench.tritonbench_g.matmul_kernel.Proofs.Native
namespace VeriTile.Bench.Repairs.MatmulKernel.Physical
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
    load (store m r a value)=fun q b => if q=r ∧ b=a then value else load m q b := by
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
    · simp [load,store,div,ha]
  · simp [load,store,hr]
theorem original_bytes (m : Bytes) (r : Region) (a : Nat) (i : Fin 4) :
    encode (load m r a) i=m r (4*a+i.val) :=
  congrFun (encode_decode (fun i => m r (4*a+i.val))) i
/-- Current-byte loads, all ordered K instructions, FP16 narrowing then FP32
widening, and an aligned four-byte write. -/
def byteStep (c : Config) (p : Instructions) (w : Work) (m : Bytes) :=
  store m .c (c.C.addr w.1 w.2) (value c (wordOps p) (load m) w)
theorem byteStep_load (c : Config) (p : Instructions) (w : Work) (m : Bytes) :
    load (byteStep c p w m)=step c (wordOps p) w (load m) := by
  unfold byteStep
  rw [load_store]
  rfl
def byteExecute (c : Config) (p : Instructions) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => byteExecute c p ws (byteStep c p w m)
theorem byteExecute_load (c : Config) (p : Instructions) (ws : List Work) (m : Bytes) :
    load (byteExecute c p ws m)=execute c (wordOps p) ws (load m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change load (byteExecute c p ws (byteStep c p w m))=_
    rw [ih,byteStep_load]
    rfl
def native (c : Config) (p : Instructions) (m : Bytes) := byteExecute c p (schedule c) m
theorem native_source (c : Config) (p : Instructions) (m : Bytes) :
    load (native c p m)=source c (wordOps p) (load m) := by
  rw [source_run]
  exact byteExecute_load c p _ m
theorem native_result (c : Config) (hc : c.Valid) (p : Instructions) (m : Bytes)
    (w : Work) (hw : c.Active w) :
    load (native c p m) .c (c.C.addr w.1 w.2)=reference c p (load m) w.1 w.2 := by
  rw [native_source]
  exact native_source_readback c hc p (load m) w hw
theorem native_bytes (c : Config) (hc : c.Valid) (p : Instructions) (m : Bytes)
    (w : Work) (hw : c.Active w) (i : Fin 4) :
    native c p m .c (4*c.C.addr w.1 w.2+i.val)=encode (reference c p (load m) w.1 w.2) i := by
  rw [←original_bytes (native c p m) .c (c.C.addr w.1 w.2) i,native_result c hc p m w hw]
theorem native_frame (c : Config) (p : Instructions) (m : Bytes) (r : Region) (b : Nat)
    (outside : ∀ w,c.Active w → ¬Writes c w r (b/4)) : native c p m r b=m r b := by
  have h := run_frame c (wordOps p) (load m) r (b/4) outside
  change execute c (wordOps p) (schedule c) (load m) r (b/4)=_ at h
  rw [←byteExecute_load c p (schedule c) m] at h
  have hb := congrArg (fun x => encode x ⟨b%4,Nat.mod_lt _ (by decide)⟩) h
  dsimp only at hb
  rw [original_bytes,original_bytes] at hb
  have eq : 4*(b/4)+b%4=b := by omega
  simpa [eq,native] using hb

theorem native_readonly (c : Config) (p : Instructions) (m : Bytes) (r : Region) (b : Nat) (h : r≠.c) :
    native c p m r b=m r b := by
  apply native_frame
  intro w _
  simp [Writes,h]
theorem native_local_transport (c : Config) (actual target : Instructions) (h : Obligations actual target)
    (m : Bytes) : native c actual m=native c target m := by
  have stepEq (w : Work) (m : Bytes) : byteStep c actual w m=byteStep c target w m := by
    simp only [byteStep,value_transport c actual target h]
  have execEq (ws : List Work) (m : Bytes) : byteExecute c actual ws m=byteExecute c target ws m := by
    induction ws generalizing m with
    | nil => rfl
    | cons w ws ih => simp only [byteExecute,stepEq,ih]
  exact execEq _ m
#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean native_source
#axiomsClean native_bytes
#axiomsClean native_frame
#axiomsClean native_local_transport
end VeriTile.Bench.Repairs.MatmulKernel.Physical
