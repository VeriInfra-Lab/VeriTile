import bench.tritonbench_g.block_sparse_attn.PhysicalAdapter

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
namespace PhysicalKernel

def arguments (p : Parameters) (batches : Nat) (access : Access α Store Byte)
    (buffers : Physical.Buffers) (m : Store) (q k v : Host.Tensor) : Host.Arguments where
  layout := layoutOf p batches (metadata access (buffers.input .rows) m)
    (metadata access (buffers.input .cols) m)
  q := q
  k := k
  v := v
  rows := (buffers.input .rows).storage
  cols := (buffers.input .cols).storage
  rowAllocation := 0
  colAllocation := 0
  scale := 0

/-- Ordinary kernel ABI for an existing destination. CSR guards inspect the
actual current INT32 cells; no allocator, dispatch, returned object, source
binding, computation, or expected-result equality is an admission field. -/
structure ABI (p : Parameters) (batches : Nat) (access : Access α Store Byte)
    (buffers : Physical.Buffers) (m : Store) (q k v out : Host.Tensor)
    (rowCount colCount : Nat) : Prop where
  header : Host.Header (arguments p batches access buffers m q k v)
  csr : CheckedCSR.Guard (arguments p batches access buffers m q k v) rowCount colCount
  inputs : (buffers.input .query).storage=q.storage ∧ (buffers.input .key).storage=k.storage ∧
    (buffers.input .value).storage=v.storage
  outputStorage : buffers.output.storage=out.storage
  outputShape : out.batches=batches ∧ out.heads=p.heads ∧ out.rows=p.length ∧ out.dim=16*p.blocks
  outputStrides : out.strides=Host.contiguous (arguments p batches access buffers m q k v).layout.shape
  parameters : p.ob=p.heads*p.length*(16*p.blocks) ∧ p.oh=p.length*(16*p.blocks) ∧ p.om=16*p.blocks
  outputExtent : 4*(out.storage.offset+batches*p.heads*p.length*(16*p.blocks))≤out.storage.bytes
  outputKind : out.fp32=Bool.true ∧ out.cuda=Bool.true ∧ out.materialized=Bool.true
  device : out.device=q.device
  backing : (∀ r,(buffers.input r).base+(buffers.input r).storage.bytes≤2^64) ∧
    buffers.output.base+buffers.output.storage.bytes≤2^64
  aligned : (∀ r,(Physical.inputAddress (buffers.input r) 0)%4=0) ∧
    (Physical.inputAddress buffers.output 0)%4=0
  separate : ∀ r,buffers.output.base+buffers.output.storage.bytes≤(buffers.input r).base ∨
    (buffers.input r).base+(buffers.input r).storage.bytes≤buffers.output.base

def Program.Valid (p : Parameters) (batches : Nat) (program : Program) : Prop :=
  program.row<ceil p.length 16 ∧ program.head<batches*p.heads ∧ program.channel<p.blocks

def Program.lane (program : Program) (i : TileIndex [16,16]) : Lane :=
  ⟨program.row,program.head,program.channel,i.1.val,i.2.1.val⟩

theorem lane_valid (p : Parameters) (batches : Nat) (rows cols : Int → Int)
    (program : Program) (valid : program.Valid p batches) (i : TileIndex [16,16]) :
    (program.lane i).Valid (layoutOf p batches rows cols).shape :=
  ⟨valid.1,valid.2.1,valid.2.2,i.1.isLt,i.2.1.isLt⟩

theorem first_nonnegative {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (m : Store)
    (q k v out : Host.Tensor) (rowCount colCount : Nat)
    (abi : ABI p batches access buffers m q k v out rowCount colCount)
    (program : Program) (valid : program.Valid p batches) :
    0≤first p (state access zero buffers m program) := by
  let s := state access zero buffers m program
  let a := arguments p batches access buffers m q k v
  let lane := program.lane (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)
  have validLane : lane.Valid a.layout.shape := lane_valid p batches _ _ program valid _
  obtain ⟨_,hh,hr,_⟩ := CheckedCSR.lane_bounds a abi.header lane validLane
  have nonnegative := (abi.csr.endpoints _ hh _ hr).1
  rw [first_view p batches s (Physical.inputs access.toAPI zero buffers m) _ _
    (state_views access zero buffers m program) ⟨0,by decide⟩]
  exact nonnegative

theorem column_nonnegative {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (m : Store)
    (q k v out : Host.Tensor) (rowCount colCount : Nat)
    (abi : ABI p batches access buffers m q k v out rowCount colCount)
    (program : Program) (valid : program.Valid p batches) (visit : Nat)
    (live : visit<visitCount p (state access zero buffers m program)) :
    0≤csrColumn p (state access zero buffers m program)
      (first p (state access zero buffers m program)+visit) := by
  let s := state access zero buffers m program
  let a := arguments p batches access buffers m q k v
  let lane := program.lane (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)
  have validLane : lane.Valid a.layout.shape := lane_valid p batches _ _ program valid _
  obtain ⟨_,hh,hr,_⟩ := CheckedCSR.lane_bounds a abi.header lane validLane
  have count := count_view p batches s (Physical.inputs access.toAPI zero buffers m) _ _
    (state_views access zero buffers m program) ⟨0,by decide⟩
  rw [count] at live
  have nonnegative := (abi.csr.columns _ hh _ hr visit live).1
  rw [column_view p batches s (Physical.inputs access.toAPI zero buffers m) _ _
    (state_views access zero buffers m program)
    (first_nonnegative access zero p batches buffers m q k v out rowCount colCount abi program valid)
    ⟨0,by decide⟩ visit]
  exact nonnegative

end PhysicalKernel
#axiomsClean PhysicalKernel.lane_valid
#axiomsClean PhysicalKernel.first_nonnegative
#axiomsClean PhysicalKernel.column_nonnegative
#specNonCircular PhysicalKernel.ABI avoiding
  [execution,output,accumulation,update,Host.invoke,Physical.execute,Spec.output,rowValue]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
