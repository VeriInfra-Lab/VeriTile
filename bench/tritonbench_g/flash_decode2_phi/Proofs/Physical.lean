import bench.tritonbench_g.flash_decode2_phi.Proofs.NativeHost

namespace VeriTile.Bench.Repairs.FlashDecode2.Physical
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

/-- Only the symbolic output view is projected. Lengths remain INT32 cells;
FP32 aliases are read through their bound allocation and never reinterpreted as
length cells. Actual caller bytes below are updated by concrete stores. -/
def projectO (I : Instructions) (S : Schedule) (s : State Term) (b : Nat) : Byte :=
  encode (observe I S s "O" (b/4)) ⟨b%4,Nat.mod_lt _ (by decide)⟩

theorem original_bytes (m : Bytes) (r : RegionName) (a : Nat) (i : Fin 4) :
    encode (load m r a) i=m r (4*a+i.val) :=
  congrFun (encode_decode (fun i => m r (4*a+i.val))) i

theorem entry_related (I : Instructions) (S : Schedule) (bindings : Bindings) (m : Bytes) :
    ∀ b,projectO I S (entry bindings (load m)) b=m bindings.out b := by
  intro b
  simp only [projectO,observe,entry,show ("O" : RegionName)≠"L" by decide,ite_false,
    Cell.read_same,eval,Bindings.at]
  simp only [show ("O" : RegionName)≠"M" by decide,show ("O" : RegionName)≠"E" by decide,
    ite_false,ite_true]
  rw [original_bytes]
  congr 1
  change 4*(b/4)+b%4=b
  omega

theorem write_related (I : Instructions) (S : Schedule) (s : State Term) (m : Bytes)
    (O : RegionName) (a : Nat) (v : Term) (h : ∀ b,projectO I S s b=m O b) :
    ∀ b,projectO I S (s.write "O" a (.mk .real v)) b=store m O a (eval I S v) O b := by
  intro b
  by_cases same : b/4=a
  · simp [projectO,observe,State.write,store,same,Cell.read_same]
  · simpa [projectO,observe,State.write,store,same] using h b

def scatter {ι : Type} (xs : List ι) (O : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term)
    (I : Instructions) (S : Schedule) (m : Bytes) : Bytes :=
  xs.foldl (fun b i => if active i then store b O (off i) (eval I S (v i)) else b) m

theorem scatter_related {ι : Type} (xs : List ι) (O : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term)
    (I : Instructions) (S : Schedule) (s : State Term) (m : Bytes)
    (h : ∀ b,projectO I S s b=m O b) :
    ∀ b,projectO I S (xs.foldl (fun st i => if active i then st.write "O" (off i)
      (.mk .real (v i)) else st) s) b=scatter xs O off active v I S m O b := by
  induction xs generalizing s m with
  | nil => exact h
  | cons i xs ih =>
    simp only [List.foldl_cons,scatter,List.foldl_cons]
    by_cases hi : active i
    · simp only [hi,ite_true]
      exact ih _ _ (write_related I S s m O (off i) (v i) h)
    · simp only [hi,ite_false]
      exact ih s m h

def executeBytes (I : Instructions) (S : Schedule) (O : RegionName) (q : Host.Request) :
    List Nat → State Term → Bytes → Bytes
  | [],_,m => m
  | row::rest,s,m => executeBytes I S O q rest
      (Grid.block algebra s q.g.masked q.g.batches q.g.heads q.g.capacity q.g.dimension
        q.g.sequenceBlock q.g.laneBlock row)
      (scatter (TileShape.allIndices [q.g.laneBlock]) O
        (fun k => row*q.g.dimension+lane k) (active q.g.masked q.g.dimension)
        (Grid.value algebra s q.g.masked q.g.batches q.g.heads q.g.capacity q.g.dimension
          q.g.sequenceBlock q.g.laneBlock row) I S m)

theorem grid_related (I : Instructions) (S : Schedule) (O : RegionName) (q : Host.Request)
    (rows : List Nat) (s : State Term) (m : Bytes) (h : ∀ b,projectO I S s b=m O b) :
    ∀ b,projectO I S (Grid.grid algebra q.g.masked q.g.batches q.g.heads q.g.capacity
      q.g.dimension q.g.sequenceBlock q.g.laneBlock rows s) b=executeBytes I S O q rows s m O b := by
  induction rows generalizing s m with
  | nil => exact h
  | cons row rest ih =>
    rw [Grid.grid,executeBytes]
    apply ih
    unfold Grid.block VeriTile.Bench.Repairs.FlashDecode2.scatter
    simp only [Grid.output_offset]
    exact scatter_related _ O _ _ _ I S _ m h

theorem scatter_frame {ι : Type} (xs : List ι) (O : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term)
    (I : Instructions) (S : Schedule) (m : Bytes) (r : RegionName) (b : Nat)
    (miss : r≠O ∨ ∀ i∈xs,active i → b/4≠off i) :
    scatter xs O off active v I S m r b=m r b := by
  induction xs generalizing m with
  | nil => rfl
  | cons i xs ih =>
    simp only [scatter,List.foldl_cons]
    have tail : r≠O ∨ ∀ j∈xs,active j → b/4≠off j :=
      miss.imp_right (fun h j hj => h j (by simp [hj]))
    change scatter xs O off active v I S (if active i then store m O (off i) (eval I S (v i)) else m) r b=m r b
    rw [ih _ tail]
    by_cases hi : active i
    · simp only [hi,ite_true,store]
      rw [if_neg]
      rcases miss with h|h
      · exact fun eq => h eq.1
      · exact fun eq => h i (by simp) hi eq.2
    · simp [hi]

theorem execute_frame (I : Instructions) (S : Schedule) (O : RegionName) (q : Host.Request)
    (rows : List Nat) (bounded : ∀ row∈rows,row<q.g.rows) (s : State Term) (m : Bytes)
    (r : RegionName) (b : Nat) (outside : r≠O ∨ 4*q.g.size≤b) :
    executeBytes I S O q rows s m r b=m r b := by
  induction rows generalizing s m with
  | nil => rfl
  | cons row rest ih =>
    rw [executeBytes,ih (fun r hr => bounded r (by simp [hr]))]
    apply scatter_frame
    rcases outside with h|h
    · exact .inl h
    · right
      intro k _ hk eq
      have bound := output_offset_bound q.g row (lane k) (bounded row (by simp))
        ((Grid.active_iff q.g.masked q.g.dimension q.g.laneBlock q.valid.2.2.1 k).mp hk)
      omega

/-- Whole source, whole host, and explicit physical byte stores. No completed
output-memory materialization is assumed. Every store in executeBytes is an
aligned little-endian four-byte mutation at the actual supplied allocation. -/
theorem whole_bytes (I : Instructions) (S : Schedule) (q : Host.Request) (memory : Bytes)
    (call : Host.Preparation Term q) (hbefore : call.before=entry (bindings call) (load memory))
    (schedule : List Nat) (all : schedule.Perm (List.range q.g.rows)) :
    ∃ result,Host.invoke algebra q call schedule=some result ∧ result.returned=none ∧
      result.suppliedOutput=call.out ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.g.rows=0 then none else some call.owner) ∧
      (∀ row (k : TileIndex [q.g.laneBlock]),row<q.g.rows → lane k<q.g.dimension → ∀ i : Fin 4,
        executeBytes I S call.out.allocation q schedule call.before memory call.out.allocation
          (4*(row*q.g.dimension+lane k)+i.val)=
          encode (Native.value I S (bindings call) (load memory) q row k) i) ∧
      (∀ r b,r≠call.out.allocation ∨ 4*q.g.size≤b →
        executeBytes I S call.out.allocation q schedule call.before memory r b=memory r b) ∧
      (∀ b,projectO I S result.state b=
        executeBytes I S call.out.allocation q schedule call.before memory call.out.allocation b) := by
  obtain ⟨result,he,hn,ho,hctx,hl,hv,hf⟩ := whole_native I S q (load memory) call hbefore schedule all
  obtain ⟨t,ht,hm⟩ := Grid.source_complete algebra q.g.masked q.g.batches q.g.heads q.g.capacity
    q.g.dimension q.g.sequenceBlock q.g.laneBlock schedule call.before
  have result_eq : result=Host.finish call t := by
    have hh : Host.invoke algebra q call schedule=some (Host.finish call t) := by simp [Host.invoke,ht]
    exact Option.some.inj (he.symm.trans hh)
  have related : ∀ b,projectO I S result.state b=
      executeBytes I S call.out.allocation q schedule call.before memory call.out.allocation b := by
    intro b
    rw [result_eq]
    change projectO I S t b=_
    unfold projectO observe
    rw [hm]
    exact grid_related I S call.out.allocation q schedule call.before memory
      (by rw [hbefore]; exact entry_related I S (bindings call) memory) b
  refine ⟨result,he,hn,ho,hctx,hl,?_,?_,related⟩
  · intro row k hr hk i
    rw [←related]
    have hd : (4*(row*q.g.dimension+lane k)+i.val)/4=row*q.g.dimension+lane k := by omega
    have hrm : (4*(row*q.g.dimension+lane k)+i.val)%4=i.val := by omega
    simp only [projectO,hd,hv row k hr hk]
    congr 1
    exact Fin.ext hrm
  · intro r b outside
    exact execute_frame I S call.out.allocation q schedule
      (fun row hr => List.mem_range.mp (all.mem_iff.mp hr)) call.before memory r b outside

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean write_related
#axiomsClean grid_related
#axiomsClean execute_frame
#axiomsClean whole_bytes
#specNonCircular store avoiding [Host.invoke,Grid.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FlashDecode2.Physical
