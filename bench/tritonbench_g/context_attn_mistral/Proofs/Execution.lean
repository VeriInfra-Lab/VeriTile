import bench.tritonbench_g.context_attn_mistral.Proofs.Checked
import bench.tritonbench_g.context_attn_mistral.Proofs.Recurrence

/-! Dynamic physical stores of the selected scalar graph. Each event reads
current memory. Checked separation proves all reads equal initial inputs;
unique packed coordinates prove exact readback and the full byte complement. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistral.Execution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Checked NativeIndex
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

abbrev Index (c : Host.Call) := Fin c.batches × Fin c.heads × Fin (c.blocks*128) × Fin c.layout.D
abbrev Coord (c : Host.Call) := {q : Index c // q.2.2.1.val<c.length q.1.val}
def batch (q : Coord c) := q.val.1.val
def head (q : Coord c) := q.val.2.1.val
def row (q : Coord c) := q.val.2.2.1.val
def dim (q : Coord c) := q.val.2.2.2

def all (c : Host.Call) : List (Coord c) := (Finset.univ : Finset (Coord c)).toList
theorem coverage (q : Coord c) : q∈all c := by simp [all]

def offset (q : Coord c) := c.output (batch q) (head q) (row q) (dim q)
def address (buffers : Buffers) (q : Coord c) := (buffers.data .output).base+4*offset q

theorem offset_bound (g : Geometry c tokens) (q : Coord c) : offset q<size c tokens .output :=
  coordinate_bound g .output _ _ _ _ q.val.1.isLt q.val.2.1.isLt q.property (dim q).isLt

theorem offset_injective (g : Geometry c tokens) : Function.Injective (@offset c) := by
  intro q p eq
  change elementOffset c .output (batch q) (head q) (row q) (dim q)=
    elementOffset c .output (batch p) (head p) (row p) (dim p) at eq
  rw [offset_dense g,offset_dense g] at eq
  obtain ⟨token,heads,dims⟩ := linear_injective _ _ _ _ _ _ _ _ q.val.2.1.isLt p.val.2.1.isLt
    (dim q).isLt (dim p).isLt eq
  obtain ⟨batches,rows⟩ := packed_unique g _ _ _ _ q.val.1.isLt p.val.1.isLt q.property p.property token
  exact Subtype.ext (Prod.ext (Fin.ext batches) (Prod.ext (Fin.ext heads) (Prod.ext (Fin.ext rows) (Fin.ext dims))))

theorem output_apart (g : Geometry c tokens) (buffers : Buffers) {q p : Coord c} (h : q≠p) :
    Apart (address buffers q) (address buffers p) := by
  have ne : offset q≠offset p := fun e=>h (offset_injective g e)
  unfold address Apart
  omega

/-- Out-of-shape coordinates are irrelevant and return a fixed zero. Every
source load used by the graph has a checked in-shape coordinate. -/
def inputRead (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat)
    (buffers : Buffers) (m : Store) (role : DataRole) (r h d : Nat) : α :=
  if r<tokens ∧ h<headsFor c role ∧ d<c.layout.D then
    api.read m ((buffers.data role).base+4*linear (headsFor c role) c.layout.D r h d)
  else zero

def inputs (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat)
    (buffers : Buffers) (m : Store) : Recurrence.Inputs α :=
  ⟨inputRead api zero c tokens buffers m .query,
   inputRead api zero c tokens buffers m .key,inputRead api zero c tokens buffers m .value⟩

def step (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat) (buffers : Buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (m : Store) (q : Coord c) : Store :=
  api.write m (address buffers q) (compute (inputs api zero c tokens buffers m) q)

def execute (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat) (buffers : Buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (order : List (Coord c)) (m : Store) : Store :=
  order.foldl (step api zero c tokens buffers compute) m

theorem read_step (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (m : Store) (q : Coord c)
    (role : DataRole) (hr : role≠.output) (r h d : Nat) :
    inputRead api zero c tokens buffers (step api zero c tokens buffers compute m q) role r h d=
      inputRead api zero c tokens buffers m role r h d := by
  unfold inputRead
  split_ifs with live
  · apply api.read_apart
    have qo := offset_bound a.geometry q
    have ri := linear_bound r h d tokens (headsFor c role) c.layout.D live.1 live.2.1 live.2.2
    have separate := a.separate role hr
    rw [a.extents,a.extents] at separate
    change linear (headsFor c role) c.layout.D r h d<size c tokens role at ri
    unfold address Apart
    omega
  · rfl

theorem inputs_step (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (m : Store) (q : Coord c) :
    inputs api zero c tokens buffers (step api zero c tokens buffers compute m q)=
      inputs api zero c tokens buffers m := by
  unfold inputs
  congr 1 <;> funext r h d
  · exact read_step api zero a compute m q .query (by decide) r h d
  · exact read_step api zero a compute m q .key (by decide) r h d
  · exact read_step api zero a compute m q .value (by decide) r h d

def event (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat) (buffers : Buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (m : Store) (q : Coord c) : Write α :=
  ⟨address buffers q,compute (inputs api zero c tokens buffers m) q⟩

theorem execute_static_aux (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (order : List (Coord c)) (initial m : Store)
    (stable : inputs api zero c tokens buffers m=inputs api zero c tokens buffers initial) :
    execute api zero c tokens buffers compute order m=
      run api (order.map (event api zero c tokens buffers compute initial)) m := by
  induction order generalizing m with
  | nil => rfl
  | cons q qs ih =>
    change execute api zero c tokens buffers compute qs (step api zero c tokens buffers compute m q)=
      run api (qs.map (event api zero c tokens buffers compute initial)) (api.write m _ _)
    rw [ih _ ((inputs_step api zero a compute m q).trans stable)]
    simp only [step,event,stable]

theorem execute_static (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (order : List (Coord c)) (m : Store) :
    execute api zero c tokens buffers compute order m=run api (order.map (event api zero c tokens buffers compute m)) m :=
  execute_static_aux api zero a compute order m m rfl

theorem readback (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (order : List (Coord c)) (m : Store)
    (q : Coord c) (present : q∈order) :
    api.read (execute api zero c tokens buffers compute order m) (address buffers q)=
      compute (inputs api zero c tokens buffers m) q := by
  rw [execute_static api zero a]
  apply run_readback
  · exact ⟨event api zero c tokens buffers compute m q,List.mem_map.mpr ⟨q,present,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨p,hp,rfl⟩ := List.mem_map.mp he
    by_cases eq : p=q
    · subst p;exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (output_apart a.geometry buffers eq)

/-- The frame is the precise complement of actual store windows: it includes
packed gaps, underlaunched rows, all reads, metadata and unrelated arena bytes. -/
def Outside (buffers : Buffers) (order : List (Coord c)) (i : Nat) : Prop :=
  ∀ q∈order,i<address buffers q ∨ address buffers q+4 ≤ i

theorem byte_frame (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (order : List (Coord c)) (m : Store)
    (i : Nat) (outside : Outside buffers order i) :
    api.byte (execute api zero c tokens buffers compute order m) i=api.byte m i := by
  rw [execute_static api zero a]
  apply run_byte_frame
  intro e he
  obtain ⟨q,hq,rfl⟩ := List.mem_map.mp he
  exact outside q hq

theorem word_finite (a : Admission c tokens request buffers) (q : Coord c) (j : Nat) (hj : j<4) :
    address buffers q+j<2^64 ∧
      (BitVec.ofNat 64 (address buffers q+j)).toNat=address buffers q+j := by
  have hb := a.backing .output
  have ho := offset_bound a.geometry q
  rw [a.extents] at hb
  have bound : address buffers q+j<2^64 := by unfold address;omega
  exact ⟨bound,by change _%2^64=_;exact Nat.mod_eq_of_lt bound⟩

/-- The actual checked wrapper control flow returns None and retains the
supplied output object. Empty calls preserve every byte without a launch. -/
def dispatch (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat)
    (request : Request) (buffers : Buffers) (compute : Recurrence.Inputs α → Coord c → α)
    (order : List (Coord c)) (m : Store) : Except Unit (Option Unit × Tensor × Store) :=
  match validate c tokens request buffers with
  | .error _ => .error ()
  | .ok Bool.false => .ok (none,request.tensor .output,m)
  | .ok Bool.true => .ok (none,request.tensor .output,execute api zero c tokens buffers compute order m)

theorem dispatch_empty (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat)
    (request : Request) (buffers : Buffers) (compute : Recurrence.Inputs α → Coord c → α)
    (order : List (Coord c)) (m : Store) (h : validate c tokens request buffers=.ok Bool.false) :
    dispatch api zero c tokens request buffers compute order m=.ok (none,request.tensor .output,m) := by simp [dispatch,h]

theorem dispatch_launch (api : API α Store Byte) (zero : α) (c : Host.Call) (tokens : Nat)
    (request : Request) (buffers : Buffers) (compute : Recurrence.Inputs α → Coord c → α)
    (order : List (Coord c)) (m : Store) (h : validate c tokens request buffers=.ok Bool.true) :
    dispatch api zero c tokens request buffers compute order m=
      .ok (none,request.tensor .output,execute api zero c tokens buffers compute order m) := by simp [dispatch,h]

/-- Every logical source read has the original caller address; the totalized
reader's extra out-of-shape branch is unreachable for an active source load. -/
theorem inputRead_active (api : API α Store Byte) (zero : α) (g : Geometry c tokens)
    (buffers : Buffers) (m : Store) (role : DataRole) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.heads) (hr : row<c.length b) (hd : d<c.layout.D) :
    inputRead api zero c tokens buffers m role (c.start b+row) (role.head c h) d=
      api.read m ((buffers.data role).base+4*elementOffset c role b h row d) := by
  rw [inputRead,if_pos ⟨by have := g.bounds b hb;omega,head_bound g role h hh,hd⟩,offset_dense g]

def point (q : Coord c) : NativeIndex.Point c :=
  ⟨q.val.1,q.val.2.1,⟨row q/128,(Nat.div_lt_iff_lt_mul (by decide)).mpr q.val.2.2.1.isLt⟩⟩

/-- Exact connection to the original masked native output pointer and every
interior byte of its complete FP32 word. -/
theorem source_pointer (a : Admission c tokens request buffers) (q : Coord c) :
    (NativeIndex.queryPointer c .fp32 buffers .output (point q) (row q%128) (dim q)).toNat=address buffers q := by
  have hrow : row q/128*128+row q%128=row q := by omega
  have h := (NativeIndex.correspondence c .fp32 buffers (native_ready a)).queryWord
    .output (Or.inr rfl) (point q) (row q%128) (dim q) (Nat.mod_lt _ (by decide)) (dim q).isLt
    (by change row q/128*128+row q%128<c.length (batch q);rw [hrow];exact q.property)
  convert h.1 using 1
  change (buffers.data .output).base+4*c.output (batch q) (head q) (row q) (dim q)=
    (buffers.data .output).base+4*c.output (batch q) (head q) (row q/128*128+row q%128) (dim q)
  rw [hrow]

theorem metadata_frame (api : API α Store Byte) (zero : α) (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs α → Coord c → α) (order : List (Coord c)) (m : Store)
    (role : MetaRole) (b : Nat) (hb : b<c.batches) :
    api.read (execute api zero c tokens buffers compute order m) ((buffers.metadataBuffer role).base+4*b)=
      api.read m ((buffers.metadataBuffer role).base+4*b) := by
  rw [execute_static api zero a]
  apply run_read_frame
  intro e he
  obtain ⟨q,hq,rfl⟩ := List.mem_map.mp he
  have ho := offset_bound a.geometry q
  have separate := a.metaSeparate role
  rw [a.extents] at separate
  change Apart (address buffers q) ((buffers.metadataBuffer role).base+4*b)
  unfold address Apart
  omega

#axiomsClean source_pointer
#axiomsClean metadata_frame
/-- Exact little-endian FP32 storage bytes; no numerical interpretation. -/
def wordByte (word : BitVec 32) (j : Fin 4) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 2^(8*j.val))

/-- Ordinary local raw-word decoder/encoder law. It relates one read to its
four existing bytes, never a kernel result or mathematical output. -/
structure WordCodec (api : API (BitVec 32) Store (BitVec 8)) : Prop where
  bytes : ∀ m a (j : Fin 4), api.byte m (a+j.val)=wordByte (api.read m a) j

theorem output_byte (api : API (BitVec 32) Store (BitVec 8)) (codec : WordCodec api)
    (a : Admission c tokens request buffers)
    (compute : Recurrence.Inputs (BitVec 32) → Coord c → BitVec 32)
    (order : List (Coord c)) (m : Store) (q : Coord c) (present : q∈order) (j : Fin 4) :
    api.byte (execute api 0 c tokens buffers compute order m) (address buffers q+j.val)=
      wordByte (compute (inputs api 0 c tokens buffers m) q) j := by
  rw [codec.bytes,readback api 0 a _ order m q present]

#axiomsClean output_byte
#axiomsClean execute_static
#axiomsClean readback
#axiomsClean byte_frame
#axiomsClean word_finite
end
end VeriTile.Bench.Repairs.ContextAttnMistral.Execution
