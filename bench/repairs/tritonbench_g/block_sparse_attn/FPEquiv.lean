import bench.repairs.tritonbench_g.block_sparse_attn.Contract
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Host
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Bounds
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Reduction
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.SourceBinding
namespace VeriTile.Bench.Repairs.BlockSparseAttn
structure WholeNative (a : Host.Arguments) (out : Host.Tensor) (caller : Nat)
    (actual reference : Native.Instructions) (initial : Native.Bytes) : Prop where
  identity : (Host.invoke a out caller actual initial).tensor=out
  shape : out.batches=a.layout.shape.batches ∧ out.heads=a.layout.shape.heads ∧
    out.rows=a.layout.shape.length ∧ out.dim=a.layout.shape.dim
  device : (Host.invoke a out caller actual initial).tensor.device=a.q.device
  launchDevice : (Host.invoke a out caller actual initial).launchDevice=a.q.device
  restoredDevice : (Host.invoke a out caller actual initial).restoredDevice=caller
  output : ∀ w : Work,w.active a.layout.shape → ∀ i : Fin 4,
    (Host.invoke a out caller actual initial).bytes .o (4*address out.strides w.batch w.head w.row w.col+i.val)=
      Native.encode (rowValue a.layout (Native.ops reference) (Native.load initial) a.scale w) i
  frame : ∀ r b,¬footprint a.layout.shape r (b/4) →
    (Host.invoke a out caller actual initial).bytes r b=initial r b

/-- Parametric native graph theorem: `a.layout` is the already resolved exact
source-specific compiler layout, under the ordinary lowering/ABI boundary.
The theorem does not certify that an arbitrary finite tree is this source's
compiler output. `whole_sm90_b16` exposes the retained concrete profile. -/
theorem whole_fp (a : Host.Arguments) (valid : Host.Contract a) (out : Host.Tensor)
    (allocation : Host.Allocation a out) (caller : Nat) (actual reference : Native.Instructions)
    (numeric : Native.Correspondence actual reference) (initial : Native.Bytes)
    (metadata : Host.MetadataBound a initial) : WholeNative a out caller actual reference initial := by
  exact ⟨rfl,allocation.shape,allocation.device,rfl,rfl,
    Host.output a valid out allocation caller actual reference numeric initial metadata,
    Host.frame a valid out caller actual initial metadata⟩

theorem whole_native (a : Host.Arguments) (valid : Host.Contract a) (out : Host.Tensor)
    (allocation : Host.Allocation a out) (caller : Nat) (native : Native.Instructions)
    (initial : Native.Bytes) (metadata : Host.MetadataBound a initial) :
    WholeNative a out caller native native initial :=
  whole_fp a valid out allocation caller native native (Native.reflexive native) initial metadata

theorem whole_sm90_b16 (a : Host.Arguments) (valid : Host.Contract a)
    (_binding : SM90B16.Binding a.layout) (out : Host.Tensor) (allocation : Host.Allocation a out)
    (caller : Nat) (actual reference : Native.Instructions) (numeric : Native.Correspondence actual reference)
    (initial : Native.Bytes) (metadata : Host.MetadataBound a initial) :
    WholeNative a out caller actual reference initial :=
  whole_fp a valid out allocation caller actual reference numeric initial metadata

#axiomsClean whole_fp
#axiomsClean whole_native
#axiomsClean whole_sm90_b16
#specNonCircular Host.MetadataBound avoiding [execute,rowValue,Host.invoke]

namespace CheckedFP
open Physical Wrapper
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Local observed signed metadata binding, with no result premise. The
underlying raw views may alias Q/K/V or each other. -/
structure MetadataBinding (api : API Word Store Byte) (a : Host.Arguments) (buffers : Buffers) (m : Store) : Prop where
  rows : ∀ (i : Nat),Safety.Allocated (buffers.input .rows).storage (i:Int) →
    (api.read m (inputAddress (buffers.input .rows) i)).toInt=a.layout.csr.rows (i:Int)
  cols : ∀ (i : Nat),Safety.Allocated (buffers.input .cols).storage (i:Int) →
    (api.read m (inputAddress (buffers.input .cols) i)).toInt=a.layout.csr.cols (i:Int)

def source (a : Host.Arguments) (native : Native.Instructions)
    (binding : SourceBinding.Binding a.layout (Native.ops native)) (m : Memory Word) (w : Work) : Word :=
  SourceBinding.value binding m a.scale w

/-- Complete initial-word recurrence. It has no dynamic execution or output
readback dependency and retains exceptional IEEE words and all-masked NaNs. -/
def target (a : Host.Arguments) (native : Native.Instructions) (m : Memory Word) (w : Work) : Word :=
  rowValue a.layout (Native.ops native) m a.scale w

theorem source_target (a : Host.Arguments) (actual standard : Native.Instructions)
    (laws : Native.Correspondence actual standard)
    (binding : SourceBinding.Binding a.layout (Native.ops actual)) (m : Memory Word) (w : Work) :
    source a actual binding m w=target a standard m w := by
  unfold source target
  rw [SourceBinding.value_eq,Native.correspondence_eq actual standard laws]

theorem metadata_frame (api : API Word Store Byte) (a : Host.Arguments) (native : Native.Instructions)
    (binding : SourceBinding.Binding a.layout (Native.ops native))
    (accepted : Admission a out buffers rows cols) (order : List (Coord a.layout.shape)) (m : Store)
    (metadata : MetadataBinding api a buffers m) :
    MetadataBinding api a buffers (Physical.execute api 0 buffers a.layout.shape (source a native binding) order m) := by
  have frame (r : Input) (i : Nat) (live : Safety.Allocated (buffers.input r).storage (i:Int)) :
      api.read (Physical.execute api 0 buffers a.layout.shape (source a native binding) order m)
        (inputAddress (buffers.input r) i)=api.read m (inputAddress (buffers.input r) i) := by
    have h:=input_frame api 0 accepted (source a native binding) order m r i
    have allocated : 4*((buffers.input r).storage.offset+i+1)≤(buffers.input r).storage.bytes := by
      simpa only [Int.toNat_natCast] using live.2
    simpa only [Physical.read,if_pos allocated] using h
  constructor
  · intro i hi;rw [frame .rows i hi];exact metadata.rows i hi
  · intro i hi;rw [frame .cols i hi];exact metadata.cols i hi

/-- Complete checked physical endpoint: returned fresh descriptor, raw words,
all four output bytes, exact complement frame and preserved signed metadata. -/
theorem checked_call (api : API Word Store (BitVec 8)) (codec : WordCodec api)
    (a : Host.Arguments) (out : Host.Tensor) (buffers : Buffers)
    (rowHeader colHeader : MetadataHeader) (rows cols : Nat)
    (actual standard : Native.Instructions) (laws : Native.Correspondence actual standard)
    (binding : SourceBinding.Binding a.layout (Native.ops actual))
    (launch : Bool) (accepted : validate a out buffers rowHeader colHeader rows cols=.ok launch)
    (order : List (Coord a.layout.shape)) (complete : ∀ q,q∈order) (m : Store)
    (metadata : MetadataBinding api a buffers m) :
    ∃ after,SourceBinding.selectedDispatch api 0 a out buffers rowHeader colHeader rows cols binding.toStoreMap
      (source a actual binding) order m=.ok (out,after) ∧
      ((launch=Bool.false ∧ after=m) ∨ (launch=Bool.true ∧
        (∀ q : Coord a.layout.shape,api.read after (Physical.outputAddress buffers a.layout.shape q.val)=
          target a standard (Physical.inputs api 0 buffers m) q.val) ∧
        (∀ (q : Coord a.layout.shape) (j : Fin 4),api.byte after (Physical.outputAddress buffers a.layout.shape q.val+j.val)=
          wordByte (target a standard (Physical.inputs api 0 buffers m) q.val) j) ∧
        (∀ i,Outside buffers a.layout.shape order i → api.byte after i=api.byte m i) ∧
        MetadataBinding api a buffers after)) := by
  rw [SourceBinding.dispatch_eq]
  cases launch
  · exact ⟨m,dispatch_empty _ _ _ _ _ _ _ _ _ _ _ _ accepted,Or.inl ⟨rfl,rfl⟩⟩
  · have live:=(validate_launch.mp accepted).2
    have words (q : Coord a.layout.shape) :
        api.read (Physical.execute api 0 buffers a.layout.shape (source a actual binding) order m)
          (Physical.outputAddress buffers a.layout.shape q.val)=
        target a standard (Physical.inputs api 0 buffers m) q.val := by
      rw [Physical.readback api 0 live.physical _ order m q (complete q),source_target a actual standard laws]
    refine ⟨_,dispatch_launch _ _ _ _ _ _ _ _ _ _ _ _ accepted,
      Or.inr ⟨rfl,words,?_,Physical.byte_frame api 0 live.physical _ order m,?_⟩⟩
    · intro q j;rw [codec.bytes,words q]
    · exact metadata_frame api a actual binding live.physical order m metadata

#axiomsClean source_target
#axiomsClean metadata_frame
#axiomsClean checked_call
#auditStmt checked_call
#specNonCircular target avoiding [Physical.execute,Wrapper.dispatch,SourceBinding.value,source]
#specNonCircular MetadataBinding avoiding [Physical.execute,rowValue,Host.invoke]
end
end CheckedFP
end VeriTile.Bench.Repairs.BlockSparseAttn
