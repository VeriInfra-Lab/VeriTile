import bench.tritonbench_g.mixed_sparse_attention.HostContract
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.Raw
open VeriTile Triton
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

abbrev Coord (c : Config) := Fin (c.Z*c.H) × Fin c.T × Fin c.BD
abbrev Payload (α : Type) := Role → Nat → α
def hz (q : Coord c) := q.1.val
def row (q : Coord c) := q.2.1.val
def dim (q : Coord c) := q.2.2.val
def offset (q : Coord c) := logicalAddress c (hz q) (row q) (dim q)
def address (request : Request) (q : Coord c) := (request.tensor .output).pointer+4*offset q
def all (c : Config) : List (Coord c) := (Finset.univ : Finset (Coord c)).toList
theorem coverage (q : Coord c) : q∈all c := by simp [all]

theorem offset_bound (q : Coord c) : offset q<tensorSize c :=
  dense_bound _ _ _ _ _ _ q.1.isLt q.2.1.isLt q.2.2.isLt

theorem offset_injective : Function.Injective (@offset c) := by
  intro q p he
  have dims:=block_unique c.BD ((hz q)*c.T+row q) ((hz p)*c.T+row p) (dim q) (dim p)
    q.2.2.isLt p.2.2.isLt he
  have rows:=block_unique c.T (hz q) (hz p) (row q) (row p) q.2.1.isLt p.2.1.isLt dims.1
  exact Prod.ext (Fin.ext rows.1) (Prod.ext (Fin.ext rows.2) (Fin.ext dims.2))

def inputs (api : API α Store Byte) (zero : α) (c : Config) (request : Request) (m : Store) : Payload α :=
  fun r i=>if r≠.output ∧ i<r.size c then api.read m ((request.tensor r).pointer+4*i) else zero

def step (api : API α Store Byte) (zero : α) (c : Config) (request : Request)
    (compute : Payload α → Coord c → α) (m : Store) (q : Coord c) : Store :=
  api.write m (address request q) (compute (inputs api zero c request m) q)

def execute (api : API α Store Byte) (zero : α) (c : Config) (request : Request)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store) : Store :=
  order.foldl (step api zero c request compute) m

theorem read_step (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (m : Store) (q : Coord c) (r : Role) (i : Nat) :
    inputs api zero c request (step api zero c request compute m q) r i=inputs api zero c request m r i := by
  unfold inputs
  split_ifs with live
  · apply api.read_apart
    have member : r∈inputRoles := by cases r <;> simp_all [inputRoles]
    have separate:=accepted.fresh r member
    have allocation:=accepted.storage r
    have output:=offset_bound q
    unfold Apart address
    omega
  · rfl

theorem inputs_step (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (m : Store) (q : Coord c) :
    inputs api zero c request (step api zero c request compute m q)=inputs api zero c request m := by
  funext r i
  exact read_step api zero accepted compute m q r i

def event (api : API α Store Byte) (zero : α) (c : Config) (request : Request)
    (compute : Payload α → Coord c → α) (m : Store) (q : Coord c) : Write α :=
  ⟨address request q,compute (inputs api zero c request m) q⟩

theorem execute_static_aux (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (initial m : Store)
    (stable : inputs api zero c request m=inputs api zero c request initial) :
    execute api zero c request compute order m=run api (order.map (event api zero c request compute initial)) m := by
  induction order generalizing m with
  | nil => rfl
  | cons q qs ih =>
    change execute api zero c request compute qs (step api zero c request compute m q)=
      run api (qs.map (event api zero c request compute initial)) (api.write m _ _)
    rw [ih _ ((inputs_step api zero accepted compute m q).trans stable)]
    simp only [step,event,stable]

theorem execute_static (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store) :
    execute api zero c request compute order m=run api (order.map (event api zero c request compute m)) m :=
  execute_static_aux api zero accepted compute order m m rfl

theorem inputs_execute (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store) :
    inputs api zero c request (execute api zero c request compute order m)=inputs api zero c request m := by
  induction order generalizing m with
  | nil=>rfl
  | cons q qs ih=>
    change inputs api zero c request (execute api zero c request compute qs (step api zero c request compute m q))=_
    rw [ih,inputs_step api zero accepted]

theorem output_apart (request : Request) {q p : Coord c} (different : q≠p) :
    Apart (address request q) (address request p) := by
  have ne : offset q≠offset p := fun h=>different (offset_injective h)
  unfold Apart address
  omega

theorem readback (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store) (q : Coord c) (present : q∈order) :
    api.read (execute api zero c request compute order m) (address request q)=compute (inputs api zero c request m) q := by
  rw [execute_static api zero accepted]
  apply run_readback
  · exact ⟨event api zero c request compute m q,List.mem_map.mpr ⟨q,present,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨p,hp,rfl⟩:=List.mem_map.mp he
    by_cases eq : p=q
    · subst p;exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (output_apart request eq)

def Outside (request : Request) (order : List (Coord c)) (i : Nat) : Prop :=
  ∀ q∈order, i < address request q ∨ address request q + 4 ≤ i

theorem byte_frame (api : API α Store Byte) (zero : α) (accepted : Admission c request values)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store)
    (i : Nat) (outside : Outside request order i) :
    api.byte (execute api zero c request compute order m) i=api.byte m i := by
  rw [execute_static api zero accepted]
  apply run_byte_frame
  intro e he
  obtain ⟨q,hq,rfl⟩:=List.mem_map.mp he
  exact outside q hq

theorem output_word (accepted : Admission c request values) (q : Coord c) (j : Fin 4) :
    address request q+j.val<2^64 ∧
    (BitVec.ofNat 64 (address request q+j.val)).toNat=address request q+j.val := by
  have bound:=offset_bound q
  have allocation:=accepted.storage .output
  have lane:=j.isLt
  have h : address request q+j.val<2^64 := by
    change _∧4*((request.tensor .output).storageOffset+tensorSize c)≤_∧_ at allocation
    unfold address
    omega
  exact ⟨h,by change _%2^64=_;exact Nat.mod_eq_of_lt h⟩

def dispatch (api : API α Store Byte) (zero : α) (c : Config) (request : Request) (values : Observed)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store) : Except Unit (Tensor × Store) :=
  match validate c request values with
  | .error _ => .error ()
  | .ok Bool.false => .ok (request.tensor .output,m)
  | .ok Bool.true => .ok (request.tensor .output,execute api zero c request compute order m)

theorem dispatch_empty (api : API α Store Byte) (zero : α) (c : Config) (request : Request) (values : Observed)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store)
    (accepted : validate c request values=.ok Bool.false) :
    dispatch api zero c request values compute order m=.ok (request.tensor .output,m) := by simp [dispatch,accepted]

theorem dispatch_launch (api : API α Store Byte) (zero : α) (c : Config) (request : Request) (values : Observed)
    (compute : Payload α → Coord c → α) (order : List (Coord c)) (m : Store)
    (accepted : validate c request values=.ok Bool.true) :
    dispatch api zero c request values compute order m=
      .ok (request.tensor .output,execute api zero c request compute order m) := by simp [dispatch,accepted]

def wordByte (word : BitVec 32) (j : Fin 4) : BitVec 8 := BitVec.ofNat 8 (word.toNat/2^(8*j.val))
structure WordCodec (api : API (BitVec 32) Store (BitVec 8)) : Prop where
  bytes : ∀ m a (j : Fin 4),api.byte m (a+j.val)=wordByte (api.read m a) j

#axiomsClean offset_injective
#axiomsClean execute_static
#axiomsClean readback
#axiomsClean byte_frame
#axiomsClean output_word
#axiomsClean dispatch_empty
#axiomsClean dispatch_launch
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.Raw
