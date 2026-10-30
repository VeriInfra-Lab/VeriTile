import bench.tritonbench_g.block_sparse_attn.Proofs.Safety
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Host
set_option maxHeartbeats 10000000

structure Tensor where
  batches : Nat
  heads : Nat
  rows : Nat
  dim : Nat
  strides : Strides
  storage : Safety.Storage
  device : Nat
  allocation : Nat
  identity : Nat
  fp32 : Bool
  cuda : Bool
  materialized : Bool
  inference : Bool
def Tensor.last (t : Tensor) := (t.batches-1)*t.strides.batch+(t.heads-1)*t.strides.head+
  (t.rows-1)*t.strides.row+(t.dim-1)*t.strides.dim
structure TensorGuard (t : Tensor) (s : Shape) : Prop where
  positive : 0<t.batches ∧ 0<t.heads ∧ 0<t.rows ∧ 0<t.dim
  metadata : ∀ z∈[t.batches,t.heads,t.rows,t.dim,t.strides.batch,t.strides.head,t.strides.row,t.strides.dim],0<z ∧ z<2^31
  kind : t.fp32=Bool.true ∧ t.cuda=Bool.true ∧ t.materialized=Bool.true ∧ t.inference=Bool.true
  extent : 4*(t.storage.offset+t.last+1)≤t.storage.bytes
  formed : t.last+(s.blockM+s.blockN)*t.strides.row<2^31

theorem tensor_bound (t : Tensor) (b h n d : Nat)
    (hb : b<t.batches) (hh : h<t.heads) (hn : n<t.rows) (hd : d<t.dim) :
    address t.strides b h n d≤t.last := by
  have h1:=Nat.mul_le_mul_right t.strides.batch (show b≤t.batches-1 by omega)
  have h2:=Nat.mul_le_mul_right t.strides.head (show h≤t.heads-1 by omega)
  have h3:=Nat.mul_le_mul_right t.strides.row (show n≤t.rows-1 by omega)
  have h4:=Nat.mul_le_mul_right t.strides.dim (show d≤t.dim-1 by omega)
  unfold address Tensor.last
  omega

theorem tensor_access (t : Tensor) (s : Shape) (valid : TensorGuard t s) (b h n d : Nat)
    (hb : b<t.batches) (hh : h<t.heads) (hn : n<t.rows) (hd : d<t.dim) :
    Safety.Allocated t.storage (address t.strides b h n d) := by
  have bound:=tensor_bound t b h n d hb hh hn hd
  have ext:=valid.extent
  constructor
  · omega
  · simp only [Int.toNat_natCast]
    omega

theorem tensor_formed (t : Tensor) (s : Shape) (valid : TensorGuard t s) (b h n d : Nat)
    (hb : b<t.batches) (hh : h<t.heads) (hn : n≤t.rows-1+(s.blockM+s.blockN)) (hd : d<t.dim) :
    address t.strides b h n d<2^31 := by
  have h1:=Nat.mul_le_mul_right t.strides.batch (show b≤t.batches-1 by omega)
  have h2:=Nat.mul_le_mul_right t.strides.head (show h≤t.heads-1 by omega)
  have h3:=Nat.mul_le_mul_right t.strides.row hn
  have h4:=Nat.mul_le_mul_right t.strides.dim (show d≤t.dim-1 by omega)
  have hi:=valid.formed
  unfold address Tensor.last at *
  nlinarith

structure Arguments where
  layout : Layout
  q : Tensor
  k : Tensor
  v : Tensor
  rows : Safety.Storage
  cols : Safety.Storage
  rowAllocation : Nat
  colAllocation : Nat
  scale : Word
structure Header (a : Arguments) : Prop where
  shape : a.layout.shape.Valid
  qguard : TensorGuard a.q a.layout.shape
  kguard : TensorGuard a.k a.layout.shape
  vguard : TensorGuard a.v a.layout.shape
  qshape : a.q.batches=a.layout.shape.batches ∧ a.q.heads=a.layout.shape.heads ∧
    a.layout.shape.length≤a.q.rows ∧ a.q.dim=a.layout.shape.dim
  kshape : a.k.batches=a.layout.shape.batches ∧ a.k.heads=a.layout.shape.kvHeads ∧
    a.layout.shape.length≤a.k.rows ∧ a.k.dim=a.layout.shape.dim
  vshape : a.v.batches=a.layout.shape.batches ∧ a.v.heads=a.layout.shape.kvHeads ∧
    a.layout.shape.length≤a.v.rows ∧ a.v.dim=a.layout.shape.dim
  strides : a.layout.q=a.q.strides ∧ a.layout.k=a.k.strides ∧ a.layout.v=a.v.strides
  devices : a.k.device=a.q.device ∧ a.v.device=a.q.device
  evenM : a.layout.evenM=Bool.true → ceil a.layout.shape.length a.layout.shape.blockM*a.layout.shape.blockM≤a.q.rows
  tiles : (∃ m,a.layout.shape.blockM=2^(m+4)) ∧ (∃ n,a.layout.shape.blockN=2^(n+4)) ∧
    (∃ d,a.layout.shape.blockD=2^(d+4))
  alignedD : a.layout.shape.blockD/8*8=a.layout.shape.blockD
  alignedN : a.layout.shape.blockN/8*8=a.layout.shape.blockN
  launch : ceil a.layout.shape.length a.layout.shape.blockM<2^31 ∧
    a.layout.shape.batches*a.layout.shape.heads≤65535 ∧ a.layout.shape.blocks≤65535
  outputFormed : (a.layout.shape.batches*a.layout.shape.heads*a.layout.shape.length+
    a.layout.shape.blockM)*a.layout.shape.dim<2^31

structure Contract (a : Arguments) extends Header a : Prop where
  data : Safety.DataContract a.layout a.rows a.cols a.k.storage a.v.storage

theorem row_envelope (s : Shape) (l : Lane) (hl : l.Valid s) :
    l.rowProgram*s.blockM+l.rowLane<s.length+s.blockM := by
  have hp:=hl.1
  have hr:=hl.2.2.2.1
  have hh:=Nat.div_mul_le_self (s.length+s.blockM-1) s.blockM
  have hd:=Nat.mul_le_mul_right s.blockM (show l.rowProgram+1≤ceil s.length s.blockM by omega)
  unfold ceil at hd
  have hs:=Nat.sub_le (s.length+s.blockM) 1
  nlinarith

theorem query_access (a : Arguments) (valid : Contract a) (l : Lane) (hl : l.Valid a.layout.shape)
    (db d : Nat) (hdb : db<a.layout.shape.blocks) (hd : d<a.layout.shape.blockD)
    (enabled : a.layout.evenM=Bool.true ∨ (laneWork a.layout.shape l).row<a.layout.shape.length) :
    Safety.Allocated a.q.storage (address a.layout.q (laneWork a.layout.shape l).batch
      (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row (db*a.layout.shape.blockD+d)) := by
  have hp:=valid.shape
  have hh : 0<a.layout.shape.heads := hp.2.1
  have hb : (laneWork a.layout.shape l).batch<a.q.batches := by
    rw [valid.qshape.1]
    exact (Nat.div_lt_iff_lt_mul hh).mpr hl.2.1
  have hhead : (laneWork a.layout.shape l).head<a.q.heads := by
    rw [valid.qshape.2.1]
    exact Nat.mod_lt _ hh
  have hrow : (laneWork a.layout.shape l).row<a.q.rows := by
    rcases enabled with even|checked
    · have pad:=valid.evenM even
      have hm:=Nat.mul_le_mul_right a.layout.shape.blockM
        (show l.rowProgram+1≤ceil a.layout.shape.length a.layout.shape.blockM by have:=hl.1;omega)
      have hr:=hl.2.2.2.1
      change l.rowProgram*a.layout.shape.blockM+l.rowLane<a.q.rows
      nlinarith
    · exact lt_of_lt_of_le checked valid.qshape.2.2.1
  have hcol : db*a.layout.shape.blockD+d<a.q.dim := by
    rw [valid.qshape.2.2.2]
    have ht:=Nat.mul_le_mul_right a.layout.shape.blockD (show db+1≤a.layout.shape.blocks by omega)
    unfold Shape.dim
    nlinarith
  rw [valid.strides.1]
  exact tensor_access a.q a.layout.shape valid.qguard _ _ _ _ hb hhead hrow hcol

def contiguous (s : Shape) : Strides := ⟨s.heads*s.length*s.dim,s.length*s.dim,s.dim,1⟩
structure Allocation (a : Arguments) (out : Tensor) : Prop where
  shape : out.batches=a.layout.shape.batches ∧ out.heads=a.layout.shape.heads ∧
    out.rows=a.layout.shape.length ∧ out.dim=a.layout.shape.dim
  strides : out.strides=contiguous a.layout.shape
  device : out.device=a.q.device
  kind : out.fp32=Bool.true ∧ out.cuda=Bool.true ∧ out.materialized=Bool.true
  storage : out.storage.offset=0 ∧ 4*(a.layout.shape.batches*a.layout.shape.heads*
    a.layout.shape.length*a.layout.shape.dim)≤out.storage.bytes
  fresh : ∀ id∈[a.q.allocation,a.k.allocation,a.v.allocation,a.rowAllocation,a.colAllocation],out.allocation≠id

theorem output_address (s : Shape) (w : Work) :
    address (contiguous s) w.batch w.head w.row w.col=w.target s := by
  unfold address contiguous Work.target outputAddress
  ring
theorem output_access (a : Arguments) (out : Tensor) (alloc : Allocation a out)
    (w : Work) (active : w.active a.layout.shape) : Safety.Allocated out.storage (w.target a.layout.shape) := by
  have h:=target_bound a.layout.shape w active
  have hs:=alloc.storage
  constructor
  · omega
  · simp only [Int.toNat_natCast]
    omega

structure Returned where
  tensor : Tensor
  bytes : Native.Bytes
  launchDevice : Nat
  restoredDevice : Nat
def boundLayout (a : Arguments) (m : Native.Bytes) : Layout :=
  {a.layout with csr:=Safety.decodeCSR a.layout.csr.rowStride a.layout.csr.colStride m}
def MetadataBound (a : Arguments) (m : Native.Bytes) : Prop :=
  a.layout.csr=Safety.decodeCSR a.layout.csr.rowStride a.layout.csr.colStride m
theorem bound_layout (a : Arguments) (m : Native.Bytes) (h : MetadataBound a m) :
    boundLayout a m=a.layout := by
  unfold boundLayout
  rw [←h]
noncomputable def invoke (a : Arguments) (out : Tensor) (callerDevice : Nat)
    (native : Native.Instructions) (m : Native.Bytes) : Returned :=
  ⟨out,Native.byteExecute (boundLayout a m) (Native.ops native) a.scale (Source.work a.layout.shape) m,
    a.q.device,callerDevice⟩

theorem complete (a : Arguments) (valid : Contract a) : Complete a.layout.shape (Source.work a.layout.shape) :=
  Source.work_complete a.layout.shape valid.shape.2.1 valid.shape.2.2.2.2.2.1
    valid.shape.2.2.2.2.2.2.2.1

theorem identity (a : Arguments) (out : Tensor) (callerDevice : Nat) (native : Native.Instructions) (m : Native.Bytes) :
    (invoke a out callerDevice native m).tensor=out ∧
    (invoke a out callerDevice native m).launchDevice=a.q.device ∧
    (invoke a out callerDevice native m).restoredDevice=callerDevice := ⟨rfl,rfl,rfl⟩

theorem output (a : Arguments) (valid : Contract a) (out : Tensor) (alloc : Allocation a out)
    (callerDevice : Nat) (actual reference : Native.Instructions) (numeric : Native.Correspondence actual reference)
    (m : Native.Bytes) (metadata : MetadataBound a m) (w : Work) (active : w.active a.layout.shape) (i : Fin 4) :
    (invoke a out callerDevice actual m).bytes .o
      (4*address out.strides w.batch w.head w.row w.col+i.val)=
    Native.encode (rowValue a.layout (Native.ops reference) (Native.load m) a.scale w) i := by
  rw [alloc.strides,output_address]
  simp only [invoke,bound_layout a m metadata]
  exact Native.output_bytes a.layout actual reference numeric a.scale _ (complete a valid) m w active i

theorem frame (a : Arguments) (valid : Contract a) (out : Tensor) (callerDevice : Nat)
    (native : Native.Instructions) (m : Native.Bytes) (metadata : MetadataBound a m) (r : FlashAttn.Region) (b : Nat)
    (outside : ¬footprint a.layout.shape r (b/4)) :
    (invoke a out callerDevice native m).bytes r b=m r b := by
  simp only [invoke,bound_layout a m metadata]
  exact Native.byte_frame a.layout (Native.ops native) a.scale _ (complete a valid) m r b outside

#axiomsClean tensor_bound
#axiomsClean tensor_access
#axiomsClean tensor_formed
#axiomsClean row_envelope
#axiomsClean query_access
#axiomsClean output_address
#axiomsClean output_access
#axiomsClean complete
#axiomsClean identity
#axiomsClean output
#axiomsClean frame
#specNonCircular Contract avoiding [invoke,execute,rowValue,Spec.output]
#specNonCircular Allocation avoiding [invoke,execute,rowValue,Spec.output]
end VeriTile.Bench.Repairs.BlockSparseAttn.Host
