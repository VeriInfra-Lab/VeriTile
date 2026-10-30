import bench.tritonbench_g.rope_backward_transform.Kernels
import bench.tritonbench_g.rope_backward_transform.Proofs.FP
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
def Request.tableRows (r : Request) (sine : Bool) := if sine then r.sinRows else r.cosRows
def Request.rowStride (r : Request) (sine : Bool) := if sine then r.sinRowStride else r.cosRowStride
def Request.colStride (r : Request) (sine : Bool) := if sine then r.sinColStride else r.cosColStride

end VeriTile.Bench.Repairs.RopeBackwardTransform

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev Laws := VeriTile.Torch.FP32.TensorLaws
abbrev Word := Numerics.Word

/-- Actual JIT arguments: contiguous prepared [B,S,H,D] gradients, and the
original two strided coefficient tables. Q/K row arguments are explicitly H*D
in the selected Python call, independently of singleton Tensor strides. -/
structure Call where
  geometry : Request
  q : Bool→Tensor
  tables : Bool→Tensor
  qLaws : ∀ k,Laws (q k)
  tableLaws : ∀ k,Laws (tables k)


def Call.source (c : Call) : ComputeKernel :=
  let r:=c.geometry
  Kernels.source .fp32 .fp32 "Q" "K" "C" "S"
    (r.qHeads*r.dimension) (r.kHeads*r.dimension) r.cosRowStride r.sinRowStride
    r.cosColStride r.sinColStride r.sequence r.batches r.qHeads r.kHeads r.dimension
    (r.paddedHeads Bool.false) (r.paddedHeads Bool.true) r.paddedDim
    (max (r.paddedHeads Bool.false) (r.paddedHeads Bool.true)) Bool.true

def launched (r : Request) : Prop := 0<r.rows ∧ 0<r.half ∧ (0<r.qHeads ∨ 0<r.kHeads)
instance (r : Request) : Decidable (launched r) := inferInstanceAs (Decidable (_∧_∧_))
def launchRows (r : Request) := if launched r then List.range r.rows else []
def outputLast (r : Request) (k : Bool) :=
  (r.rows-1)*(r.heads k*r.dimension)+(r.paddedHeads k-1)*r.dimension+(r.paddedDim/2-1)+r.half
def tableLast (r : Request) (sine : Bool) :=
  (r.sequence-1)*r.rowStride sine+(r.paddedDim/2-1)*r.colStride sine

def apart (a b : Tensor) : Prop :=
  a.storage.object≠b.storage.object ∧
    (a.storage.elements=0 ∨ b.storage.elements=0 ∨
      a.storage.base+4*a.storage.elements≤b.storage.base ∨
      b.storage.base+4*b.storage.elements≤a.storage.base)

/-- Existing finite metadata/address checks, plus ordinary table-index API
laws. The table law binds the two actual observed strides; it permits zero
strides, independent row/column strides, storage offsets, and readonly aliases.
No source success, output, numerical exactness, or allocation is assumed. -/
structure Admitted (c : Call) : Prop where
  qShape : ∀ k,(c.q k).metadata.shape=[c.geometry.batches,c.geometry.sequence,c.geometry.heads k,c.geometry.dimension]
  tableShape : ∀ k,(c.tables k).metadata.shape=[c.geometry.tableRows k,c.geometry.half]
  contiguous : ∀ k,(c.q k).metadata.isContiguous=Bool.true
  qResolved : ∀ k,(c.q k).metadata.isNeg=Bool.false
  tableResolved : ∀ k,(c.tables k).metadata.isNeg=Bool.false
  qDevice : ∀ k,(c.q k).metadata.device.kind="cuda" ∧ (c.q k).metadata.device=(c.q Bool.false).metadata.device
  tableDevice : ∀ k,(c.tables k).metadata.device=(c.q Bool.false).metadata.device
  rows : ∀ k,c.geometry.sequence≤c.geometry.tableRows k
  tableIndex : ∀ k row lane,row<c.geometry.tableRows k → lane<c.geometry.half →
    (c.tables k).physicalIndex (row*c.geometry.half+lane)=
      (c.tables k).storageOffset+row*c.geometry.rowStride k+lane*c.geometry.colStride k
  qApart : apart (c.q Bool.false) (c.q Bool.true)
  tableApart : ∀ k s,apart (c.q k) (c.tables s)
  alignedQ : ∀ k,0<(c.q k).metadata.numel →(c.q k).dataPtr.toNat%4=0
  alignedTable : ∀ k,0<(c.tables k).metadata.numel →(c.tables k).dataPtr.toNat%4=0
  metadataBounds : c.geometry.batches<2^31 ∧ c.geometry.rows<2^31 ∧ c.geometry.sequence<2^31 ∧
    c.geometry.dimension<2^31 ∧ c.geometry.paddedDim≤65536
  shapeBounds : ∀ k,c.geometry.heads k<2^31 ∧ c.geometry.heads k*c.geometry.dimension<2^31 ∧
    c.geometry.paddedHeads k*c.geometry.paddedDim≤65536 ∧
    c.geometry.rowStride k<2^31 ∧ c.geometry.colStride k<2^31
  formedBounds : launched c.geometry →∀ k,
    outputLast c.geometry k<2^31 ∧ tableLast c.geometry k<2^31 ∧
    (c.q k).dataPtr.toNat+4*outputLast c.geometry k+4≤2^64 ∧
    (c.tables k).dataPtr.toNat+4*tableLast c.geometry k+4≤2^64

/-- The unused batch/block constexprs have no source statements; this is the
entire readable typed JIT, identical to the retained successful execution. -/
theorem source_binding (c : Call) : c.source=Source.kernel .fp32 .fp32 (parameters c.geometry) := rfl

def realPair := Spec.pair

def nativePrimitives (cfg : FP.Config) : Numerics.Primitives where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  sub := FP.sub cfg
  mul := FP.mul cfg
  neg := FP.Value.neg
  fma := FP.fma cfg

#axiomsClean source_binding
#specNonCircular Admitted avoiding [Call.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
