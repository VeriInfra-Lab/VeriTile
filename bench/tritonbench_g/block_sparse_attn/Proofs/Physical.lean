import bench.tritonbench_g.block_sparse_attn.Proofs.CheckedCSR
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.BlockSparseAttn.Physical
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

inductive Input where
  | query | key | value | rows | cols
  deriving DecidableEq, Fintype

structure Buffer where
  base : Nat
  storage : Safety.Storage

structure Buffers where
  input : Input → Buffer
  output : Buffer

def inputAddress (buffer : Buffer) (i : Nat) := buffer.base+4*(buffer.storage.offset+i)
def outputAddress (buffers : Buffers) (s : Shape) (w : Work) := inputAddress buffers.output (w.target s)

def read (api : API α Store Byte) (zero : α) (buffer : Buffer) (m : Store) (i : Nat) : α :=
  if 4*(buffer.storage.offset+i+1)≤buffer.storage.bytes then api.read m (inputAddress buffer i) else zero

/-- Floating input roles are interpreted directly in the shared current
memory. Unused roles and unreachable out-of-allocation reads are totalized. -/
def inputs (api : API α Store Byte) (zero : α) (buffers : Buffers) (m : Store) : Memory α
  | .q => read api zero (buffers.input .query) m
  | .k => read api zero (buffers.input .key) m
  | .v => read api zero (buffers.input .value) m
  | _ => fun _=>zero

structure Admission (a : Host.Arguments) (out : Host.Tensor) (buffers : Buffers)
    (rowCount colCount : Nat) : Prop where
  header : Host.Header a
  csr : CheckedCSR.Guard a rowCount colCount
  selected : a.layout.shape.blockM=16 ∧ a.layout.shape.blockN=16 ∧ a.layout.shape.blockD=16
  allocation : Host.Allocation a out
  inputStorage : (buffers.input .query).storage=a.q.storage ∧
    (buffers.input .key).storage=a.k.storage ∧ (buffers.input .value).storage=a.v.storage ∧
    (buffers.input .rows).storage=a.rows ∧ (buffers.input .cols).storage=a.cols
  outputStorage : buffers.output.storage=out.storage
  backing : (∀ r,(buffers.input r).base+(buffers.input r).storage.bytes≤2^64) ∧
    buffers.output.base+buffers.output.storage.bytes≤2^64
  aligned : (∀ r,(inputAddress (buffers.input r) 0)%4=0) ∧ (inputAddress buffers.output 0)%4=0
  separate : ∀ r,buffers.output.base+buffers.output.storage.bytes≤(buffers.input r).base ∨
    (buffers.input r).base+(buffers.input r).storage.bytes≤buffers.output.base

theorem contract (accepted : Admission a out buffers rows cols) : Host.Contract a :=
  CheckedCSR.contract a accepted.header accepted.csr

abbrev Coord (s : Shape) := {w : Work // w.active s}
def all (s : Shape) : List (Coord s) := (schedule s).pmap (fun w h=>⟨w,(schedule_mem s w).mp h⟩) (fun _ h=>h)
theorem coverage (q : Coord s) : q∈all s := by
  simp only [all,List.mem_pmap]
  exact ⟨q.val,(schedule_mem s q.val).mpr q.property,rfl⟩

theorem output_bound (accepted : Admission a out buffers rows cols) (q : Coord a.layout.shape) :
    outputAddress buffers a.layout.shape q.val+4≤buffers.output.base+buffers.output.storage.bytes := by
  have allocated:=Host.output_access a out accepted.allocation q.val q.property
  rw [←accepted.outputStorage] at allocated
  simp only [Safety.Allocated,Int.toNat_natCast] at allocated
  unfold outputAddress inputAddress
  omega

def step (api : API α Store Byte) (zero : α) (buffers : Buffers) (s : Shape)
    (compute : Memory α → Work → α) (m : Store) (q : Coord s) : Store :=
  api.write m (outputAddress buffers s q.val) (compute (inputs api zero buffers m) q.val)

def execute (api : API α Store Byte) (zero : α) (buffers : Buffers) (s : Shape)
    (compute : Memory α → Work → α) (order : List (Coord s)) (m : Store) : Store :=
  order.foldl (step api zero buffers s compute) m

theorem read_step (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (m : Store) (q : Coord a.layout.shape) (r : Input) (i : Nat) :
    read api zero (buffers.input r) (step api zero buffers a.layout.shape compute m q) i=
      read api zero (buffers.input r) m i := by
  unfold read
  split_ifs with live
  · apply api.read_apart
    have output:=output_bound accepted q
    have separate:=accepted.separate r
    unfold Apart outputAddress inputAddress at *
    omega
  · rfl

theorem inputs_step (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (m : Store) (q : Coord a.layout.shape) :
    inputs api zero buffers (step api zero buffers a.layout.shape compute m q)=inputs api zero buffers m := by
  funext r i
  cases r <;> simp only [inputs]
  all_goals try rfl
  all_goals exact read_step api zero accepted compute m q _ i

theorem input_frame (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (order : List (Coord a.layout.shape)) (m : Store) (r : Input) (i : Nat) :
    read api zero (buffers.input r) (execute api zero buffers a.layout.shape compute order m) i=
      read api zero (buffers.input r) m i := by
  induction order generalizing m with
  | nil => rfl
  | cons q qs ih =>
    change read api zero (buffers.input r)
      (execute api zero buffers a.layout.shape compute qs (step api zero buffers a.layout.shape compute m q)) i=_
    rw [ih,read_step api zero accepted]

def event (api : API α Store Byte) (zero : α) (buffers : Buffers) (s : Shape)
    (compute : Memory α → Work → α) (m : Store) (q : Coord s) : Write α :=
  ⟨outputAddress buffers s q.val,compute (inputs api zero buffers m) q.val⟩

theorem execute_static_aux (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (order : List (Coord a.layout.shape)) (initial m : Store)
    (stable : inputs api zero buffers m=inputs api zero buffers initial) :
    execute api zero buffers a.layout.shape compute order m=
      run api (order.map (event api zero buffers a.layout.shape compute initial)) m := by
  induction order generalizing m with
  | nil => rfl
  | cons q qs ih =>
    change execute api zero buffers a.layout.shape compute qs (step api zero buffers a.layout.shape compute m q)=
      run api (qs.map (event api zero buffers a.layout.shape compute initial)) (api.write m _ _)
    rw [ih _ ((inputs_step api zero accepted compute m q).trans stable)]
    simp only [step,event,stable]

theorem execute_static (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (order : List (Coord a.layout.shape)) (m : Store) :
    execute api zero buffers a.layout.shape compute order m=
      run api (order.map (event api zero buffers a.layout.shape compute m)) m :=
  execute_static_aux api zero accepted compute order m m rfl

theorem output_apart (buffers : Buffers) {q p : Coord s} (different : q≠p) :
    Apart (outputAddress buffers s q.val) (outputAddress buffers s p.val) := by
  have ne : q.val.target s≠p.val.target s := fun he=>
    different (Subtype.ext (target_injective s q.val p.val q.property p.property he))
  unfold Apart outputAddress inputAddress
  omega

theorem readback (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (order : List (Coord a.layout.shape)) (m : Store) (q : Coord a.layout.shape) (present : q∈order) :
    api.read (execute api zero buffers a.layout.shape compute order m) (outputAddress buffers a.layout.shape q.val)=
      compute (inputs api zero buffers m) q.val := by
  rw [execute_static api zero accepted]
  apply run_readback
  · exact ⟨event api zero buffers a.layout.shape compute m q,List.mem_map.mpr ⟨q,present,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨p,hp,rfl⟩:=List.mem_map.mp he
    by_cases eq : p=q
    · subst p;exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (output_apart buffers eq)

def Outside (buffers : Buffers) (s : Shape) (order : List (Coord s)) (i : Nat) : Prop :=
  ∀ q∈order, i < outputAddress buffers s q.val ∨ outputAddress buffers s q.val + 4 ≤ i

theorem byte_frame (api : API α Store Byte) (zero : α)
    (accepted : Admission a out buffers rows cols) (compute : Memory α → Work → α)
    (order : List (Coord a.layout.shape)) (m : Store) (i : Nat) (outside : Outside buffers a.layout.shape order i) :
    api.byte (execute api zero buffers a.layout.shape compute order m) i=api.byte m i := by
  rw [execute_static api zero accepted]
  apply run_byte_frame
  intro e he
  obtain ⟨q,hq,rfl⟩:=List.mem_map.mp he
  exact outside q hq

theorem read_active (api : API α Store Byte) (zero : α) (buffer : Buffer) (m : Store)
    (i : Int) (allocated : Safety.Allocated buffer.storage i) :
    read api zero buffer m i.toNat=api.read m (inputAddress buffer i.toNat) :=
  if_pos allocated.2

/-- The totalized readers are exact current caller loads at every enabled
source Q/K/V address; no padding fallback replaces an active load. -/
theorem query_load (api : API α Store Byte) (zero : α) (accepted : Admission a out buffers rows cols)
    (m : Store) (l : Lane) (valid : l.Valid a.layout.shape) (db d : Nat)
    (hdb : db<a.layout.shape.blocks) (hd : d<a.layout.shape.blockD)
    (enabled : a.layout.evenM=Bool.true ∨ (laneWork a.layout.shape l).row<a.layout.shape.length) :
    inputs api zero buffers m .q (address a.layout.q (laneWork a.layout.shape l).batch
      (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row (db*a.layout.shape.blockD+d))=
    api.read m (inputAddress (buffers.input .query) (address a.layout.q (laneWork a.layout.shape l).batch
      (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row (db*a.layout.shape.blockD+d))) := by
  have allocated:=Host.query_access a (contract accepted) l valid db d hdb hd enabled
  rw [←accepted.inputStorage.1] at allocated
  simpa only [inputs,Int.toNat_natCast] using read_active api zero (buffers.input .query) m _ allocated

theorem key_value_load (api : API α Store Byte) (zero : α) (accepted : Admission a out buffers rows cols)
    (m : Store) (l : Lane) (valid : l.Valid a.layout.shape) (visit : Nat)
    (hv : visit<a.layout.csr.count a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row)
    (db d n : Nat) (hdb : db<a.layout.shape.blocks) (hd : d<a.layout.shape.blockD) (hn : n<a.layout.shape.blockN)
    (enabled : a.layout.evenN=Bool.true ∨ a.layout.csr.key a.layout.shape (laneWork a.layout.shape l).head
      (laneWork a.layout.shape l).row visit n<(a.layout.shape.length:Int)) :
    let w:=laneWork a.layout.shape l
    let key:=a.layout.csr.key a.layout.shape w.head w.row visit n
    let ki:=(signedAddress a.layout.k w.batch (kvHead a.layout.shape w.head) key (db*a.layout.shape.blockD+d)).toNat
    let vi:=(signedAddress a.layout.v w.batch (kvHead a.layout.shape w.head) key (l.channelProgram*a.layout.shape.blockD+d)).toNat
    inputs api zero buffers m .k ki=api.read m (inputAddress (buffers.input .key) ki) ∧
    inputs api zero buffers m .v vi=api.read m (inputAddress (buffers.input .value) vi) := by
  obtain ⟨k,v⟩:=(contract accepted).data.keys l valid visit hv db hdb d hd n hn enabled
  rw [←accepted.inputStorage.2.1] at k
  rw [←accepted.inputStorage.2.2.1] at v
  exact ⟨read_active api zero (buffers.input .key) m _ k,read_active api zero (buffers.input .value) m _ v⟩

theorem output_word (accepted : Admission a out buffers rows cols) (q : Coord a.layout.shape) (j : Fin 4) :
    outputAddress buffers a.layout.shape q.val+j.val<2^64 ∧
    (BitVec.ofNat 64 (outputAddress buffers a.layout.shape q.val+j.val)).toNat=
      outputAddress buffers a.layout.shape q.val+j.val := by
  have bound:=output_bound accepted q
  have backing:=accepted.backing.2
  have lane:=j.isLt
  have h : outputAddress buffers a.layout.shape q.val+j.val<2^64 := by omega
  exact ⟨h,by change _%2^64=_;exact Nat.mod_eq_of_lt h⟩

def wordByte (word : Word) (j : Fin 4) : BitVec 8 := BitVec.ofNat 8 (word.toNat/2^(8*j.val))
structure WordCodec (api : API Word Store (BitVec 8)) : Prop where
  bytes : ∀ m a (j : Fin 4),api.byte m (a+j.val)=wordByte (api.read m a) j

#axiomsClean contract
#axiomsClean coverage
#axiomsClean execute_static
#axiomsClean readback
#axiomsClean byte_frame
#axiomsClean query_load
#axiomsClean key_value_load
#axiomsClean output_word
#specNonCircular Admission avoiding [execute,rowValue,Host.invoke,Spec.output]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.Physical
