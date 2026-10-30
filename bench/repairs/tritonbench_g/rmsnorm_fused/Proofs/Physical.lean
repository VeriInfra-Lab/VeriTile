import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.NativeHost

namespace VeriTile.Bench.Repairs.RmsnormFused.Physical
open VeriTile Triton FP.Structural Native
noncomputable section
open Classical
abbrev Byte := BitVec 8
abbrev Bytes := RegionName → Nat → Byte

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

def load (m : Bytes) : RegionName → Nat → Word := fun r a => decode (fun i => m r (4*a+i.val))

/-- Exact little-endian aligned four-byte store. -/
def store (m : Bytes) (r : RegionName) (a : Nat) (value : Word) : Bytes :=
  fun q b => if q=r ∧ b/4=a then encode value ⟨b%4,Nat.mod_lt _ (by decide)⟩ else m q b

def project (I : Instructions) (S : Schedule) (s : State Term) : Bytes :=
  fun r b => encode (observe I S s r (b/4)) ⟨b%4,Nat.mod_lt _ (by decide)⟩

theorem original_bytes (m : Bytes) (r : RegionName) (a : Nat) (i : Fin 4) :
    encode (load m r a) i=m r (4*a+i.val) :=
  congrFun (encode_decode (fun i => m r (4*a+i.val))) i

theorem entry_bytes (I : Instructions) (S : Schedule) (m : Bytes) :
    project I S (entry (load m))=m := by
  funext r b
  change encode (load m r (b/4)) ⟨b%4,by omega⟩=m r b
  rw [original_bytes]
  congr 1
  change 4*(b/4)+b%4=b
  omega

/-- One actual typed source write commutes with one physical four-byte store.
This is proved from State.write; there is no whole-materialization premise. -/
theorem write_commutes (I : Instructions) (S : Schedule) (s : State Term)
    (r : RegionName) (a : Nat) (v : Term) :
    project I S (s.write r a (.mk .real v))=store (project I S s) r a (eval I S v) := by
  funext q b
  by_cases h : q=r ∧ b/4=a
  · simp [project,observe,State.write,store,h,Cell.read_same]
  · simp [project,observe,State.write,store,h]

def scatter {ι : Type} (xs : List ι) (Y : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term)
    (I : Instructions) (S : Schedule) (m : Bytes) : Bytes :=
  xs.foldl (fun b i => if active i then store b Y (off i) (eval I S (v i)) else b) m

theorem scatter_commutes {ι : Type} (xs : List ι) (Y : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term)
    (I : Instructions) (S : Schedule) (s : State Term) :
    project I S (xs.foldl (fun st i => if active i then st.write Y (off i) (.mk .real (v i)) else st) s)=
      scatter xs Y off active v I S (project I S s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,scatter,List.foldl_cons]
    by_cases h : active i
    · simp only [h,ite_true]
      rw [ih,write_commutes]
      rfl
    · simp only [h,ite_false]
      exact ih s

def byteBlock (I : Instructions) (R : Reductions) (S : Schedule) (s : State Term)
    (X Y W : RegionName) (row N B : Nat) (eps : ℝ) : Bytes :=
  scatter (TileShape.allIndices [B]) Y (fun i => row*N+i.1.val) (fun i => i.1.val<N)
    (fun i => Row.value (algebra R) s X W row N B eps i.1.val) I S (project I S s)

theorem block_commutes (I : Instructions) (R : Reductions) (S : Schedule) (s : State Term)
    (X Y W : RegionName) (row N B : Nat) (eps : ℝ) :
    project I S (Row.block (algebra R) s X Y W row N B eps)=byteBlock I R S s X Y W row N B eps :=
  scatter_commutes _ _ _ _ _ _ _ _

/-- A concrete byte-store fold for every row. Symbolic arithmetic carries only
the source graph; each memory mutation is the four-byte store defined above. -/
def executeBytes (I : Instructions) (R : Reductions) (S : Schedule)
    (X Y W : RegionName) (N B : Nat) (eps : ℝ) : List Nat → State Term → Bytes → Bytes
  | [],_,m => m
  | row::rest,s,m =>
    executeBytes I R S X Y W N B eps rest (Row.block (algebra R) s X Y W row N B eps)
      (scatter (TileShape.allIndices [B]) Y (fun i => row*N+i.1.val) (fun i => i.1.val<N)
        (fun i => Row.value (algebra R) s X W row N B eps i.1.val) I S m)

theorem grid_commutes (I : Instructions) (R : Reductions) (S : Schedule)
    (X Y W : RegionName) (N B : Nat) (eps : ℝ) (rows : List Nat) (s : State Term) :
    project I S (Row.grid (algebra R) X Y W N B eps rows s)=
      executeBytes I R S X Y W N B eps rows s (project I S s) := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [Row.grid,ih,block_commutes]
    rfl

theorem source_commutes (I : Instructions) (R : Reductions) (S : Schedule)
    (X Y W : RegionName) (M N B : Nat) (eps : ℝ) (hN : 0<N) (hNB : N≤B)
    (rows : List Nat) (s : State Term) :
    ∃ t,Row.source (algebra R) X Y W M N B eps rows s=some t ∧
      project I S t=executeBytes I R S X Y W N B eps rows s (project I S s) := by
  obtain ⟨t,ht,hm⟩ := Row.source_complete (algebra R) X Y W M N B eps hN hNB rows s
  refine ⟨t,ht,?_⟩
  have projection : project I S t=project I S (Row.grid (algebra R) X Y W N B eps rows s) := by
    funext r b
    unfold project observe
    rw [hm]
  exact projection.trans (grid_commutes I R S X Y W N B eps rows s)

/-- Every returned FP32 word is realized by its four physical bytes, and all
caller bytes outside the output allocation footprint remain unchanged. -/
theorem whole_bytes (I : Instructions) (R : Reductions) (S : Schedule)
    (q : Host.Request) (memory : Bytes)
    (call : Host.Preparation (State Term) (Cell Term) State.mem q)
    (hbefore : call.before=entry (load memory)) (schedule : List Nat)
    (all : schedule.Perm q.geometry.grid) :
    ∃ result,Host.invoke (algebra R) q call schedule=some result ∧
      result.tensor=call.output ∧ result.tensor.shape=q.shape ∧
      result.tensor.dtype="float32" ∧ result.tensor.device=call.input.device ∧
      result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.rows=0 then none else some call.input.device) ∧
      (∀ row k,row<q.rows → k<q.columns → ∀ i : Fin 4,
        project I S result.state result.tensor.region (4*(row*q.columns+k)+i.val)=
          encode (value I R S (load memory) call.input.region call.weight.region q row k) i) ∧
      (∀ r b,r≠result.tensor.region ∨ 4*q.size≤b →
        project I S result.state r b=memory r b) := by
  obtain ⟨result,he,ht,hs,hd,hdev,hctx,hlaunch,hv,hf⟩ :=
    whole_native I R S q (load memory) call hbefore schedule all
  refine ⟨result,he,ht,hs,hd,hdev,hctx,hlaunch,?_,?_⟩
  · intro row k hr hk i
    have div : (4*(row*q.columns+k)+i.val)/4=row*q.columns+k := by omega
    have rem : (4*(row*q.columns+k)+i.val)%4=i.val := by omega
    simp only [project,div,hv row k hr hk]
    congr 1
    exact Fin.ext rem
  · intro r b h
    have out : r≠result.tensor.region ∨ q.size≤b/4 := h.imp_right (by intro h; omega)
    unfold project
    rw [hf r (b/4) out,original_bytes]
    congr 1
    change 4*(b/4)+b%4=b
    omega

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean entry_bytes
#axiomsClean write_commutes
#axiomsClean scatter_commutes
#axiomsClean block_commutes
#axiomsClean grid_commutes
#axiomsClean source_commutes
#axiomsClean whole_bytes
#specNonCircular store avoiding [Host.invoke,Row.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RmsnormFused.Physical
