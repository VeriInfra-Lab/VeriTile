import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Physical

namespace VeriTile.Bench.Repairs.BlockSparseAttn.Wrapper
open Physical
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical

structure MetadataHeader where
  int32 : Bool
  cuda : Bool
  device : Nat
  contiguous : Bool
  materialized : Bool

def tensorKind (t : Host.Tensor) : Prop :=
  t.fp32=Bool.true ∧ t.cuda=Bool.true ∧ t.materialized=Bool.true ∧ t.inference=Bool.true

/-- Observed raw wrapper header, before any launch-only storage or metadata
content checks. The scale word is the ordinary marshalled finite FP32 scalar. -/
structure Schema (a : Host.Arguments) (rowHeader colHeader : MetadataHeader)
    (rowCount colCount : Nat) : Prop where
  tensors : tensorKind a.q ∧ tensorKind a.k ∧ tensorKind a.v
  contiguous : a.layout.shape.batches≠0 → a.layout.shape.length≠0 →
    a.q.strides.dim=1 ∧ a.k.strides.dim=1 ∧ a.v.strides.dim=1
  shape : a.q.batches=a.layout.shape.batches ∧ a.q.heads=a.layout.shape.heads ∧
    a.layout.shape.length≤a.q.rows ∧ a.q.dim=a.layout.shape.dim ∧
    a.k.batches=a.layout.shape.batches ∧ a.k.heads=a.layout.shape.kvHeads ∧
    a.layout.shape.length≤a.k.rows ∧ a.k.dim=a.layout.shape.dim ∧
    a.v.batches=a.layout.shape.batches ∧ a.v.heads=a.layout.shape.kvHeads ∧
    a.layout.shape.length≤a.v.rows ∧ a.v.dim=a.layout.shape.dim
  devices : a.k.device=a.q.device ∧ a.v.device=a.q.device
  dimensions : 0<a.layout.shape.heads ∧ 0<a.layout.shape.kvHeads ∧
    a.layout.shape.kvHeads∣a.layout.shape.heads ∧ 0<a.layout.shape.blocks ∧ 0<a.layout.shape.layouts ∧
    a.layout.shape.blockM=16 ∧ a.layout.shape.blockN=16 ∧ a.layout.shape.blockD=16
  widths : ∀ n∈[a.q.batches,a.q.heads,a.q.rows,a.q.dim,a.k.batches,a.k.heads,a.k.rows,a.k.dim,
    a.v.batches,a.v.heads,a.v.rows,a.v.dim,a.layout.shape.length,a.layout.shape.blocks,
    a.layout.shape.layouts,a.layout.csr.rowStride,a.layout.csr.colStride,rowCount,colCount],n<2^31
  strides : 0<a.layout.csr.rowStride ∧ 0<a.layout.csr.colStride
  metadata : ∀ h∈[rowHeader,colHeader],h.int32=Bool.true ∧ h.cuda=Bool.true ∧ h.device=a.q.device ∧
    h.contiguous=Bool.true ∧ h.materialized=Bool.true

structure Live (a : Host.Arguments) (out : Host.Tensor) (buffers : Buffers)
    (rowHeader colHeader : MetadataHeader) (rowCount colCount : Nat) : Prop where
  schema : Schema a rowHeader colHeader rowCount colCount
  physical : Admission a out buffers rowCount colCount

structure Empty (a : Host.Arguments) (out : Host.Tensor)
    (rowHeader colHeader : MetadataHeader) (rowCount colCount : Nat) : Prop where
  schema : Schema a rowHeader colHeader rowCount colCount
  empty : a.layout.shape.batches=0 ∨ a.layout.shape.length=0
  allocation : Host.Allocation a out

def validate (a : Host.Arguments) (out : Host.Tensor) (buffers : Buffers)
    (rowHeader colHeader : MetadataHeader) (rowCount colCount : Nat) : Except Unit Bool :=
  if a.layout.shape.batches=0 ∨ a.layout.shape.length=0 then
    if Empty a out rowHeader colHeader rowCount colCount then .ok Bool.false else .error ()
  else if Live a out buffers rowHeader colHeader rowCount colCount then .ok Bool.true else .error ()

theorem validate_launch : validate a out buffers rowHeader colHeader rows cols=.ok Bool.true ↔
    ¬(a.layout.shape.batches=0 ∨ a.layout.shape.length=0) ∧ Live a out buffers rowHeader colHeader rows cols := by
  simp only [validate]
  split_ifs <;> simp_all

theorem validate_empty : validate a out buffers rowHeader colHeader rows cols=.ok Bool.false ↔
    Empty a out rowHeader colHeader rows cols := by
  unfold validate
  by_cases empty : a.layout.shape.batches=0 ∨ a.layout.shape.length=0
  · simp [empty]
  · have no : ¬Empty a out rowHeader colHeader rows cols := fun h=>empty h.empty
    simp only [if_neg empty]
    split_ifs <;> simp [no]

/-- Exact returned allocation and memory for successful checked calls. -/
def dispatch (api : API α Store Byte) (zero : α) (a : Host.Arguments) (out : Host.Tensor)
    (buffers : Buffers) (rowHeader colHeader : MetadataHeader) (rows cols : Nat)
    (compute : Memory α → Work → α) (order : List (Coord a.layout.shape)) (m : Store) :
    Except Unit (Host.Tensor × Store) :=
  match validate a out buffers rowHeader colHeader rows cols with
  | .error _ => .error ()
  | .ok Bool.false => .ok (out,m)
  | .ok Bool.true => .ok (out,Physical.execute api zero buffers a.layout.shape compute order m)

theorem dispatch_empty (api : API α Store Byte) (zero : α) (a : Host.Arguments) (out : Host.Tensor)
    (buffers : Buffers) (rowHeader colHeader : MetadataHeader) (rows cols : Nat)
    (compute : Memory α → Work → α) (order : List (Coord a.layout.shape)) (m : Store)
    (accepted : validate a out buffers rowHeader colHeader rows cols=.ok Bool.false) :
    dispatch api zero a out buffers rowHeader colHeader rows cols compute order m=.ok (out,m) := by
  simp [dispatch,accepted]

theorem dispatch_launch (api : API α Store Byte) (zero : α) (a : Host.Arguments) (out : Host.Tensor)
    (buffers : Buffers) (rowHeader colHeader : MetadataHeader) (rows cols : Nat)
    (compute : Memory α → Work → α) (order : List (Coord a.layout.shape)) (m : Store)
    (accepted : validate a out buffers rowHeader colHeader rows cols=.ok Bool.true) :
    dispatch api zero a out buffers rowHeader colHeader rows cols compute order m=
      .ok (out,Physical.execute api zero buffers a.layout.shape compute order m) := by
  simp [dispatch,accepted]

#axiomsClean validate_launch
#axiomsClean validate_empty
#axiomsClean dispatch_empty
#axiomsClean dispatch_launch
#specNonCircular Live avoiding [Host.invoke,execute,rowValue,Spec.output,dispatch]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.Wrapper
