import bench.repairs.tritonbench_g.rope_transform.Kernels
import bench.repairs.tritonbench_g.rope_transform.Proofs.Grid
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
namespace Reuse
export VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
  (Tensor Laws Word logicalRead logical_word pointer_exact)
end Reuse
open Reuse
abbrev Tensor := Reuse.Tensor
abbrev Laws := Reuse.Laws
abbrev Word := Reuse.Word
abbrev logicalRead {β : Type} (read : Nat→β) (tensor : Tensor) (index : Nat) : β :=
  read (tensor.storage.address (tensor.physicalIndex index))

def RequestSize (r : Request) (key : Bool) := r.rows*r.stride key

def outputLast (r : Request) (key : Bool) :=
  (r.rows-1)*r.stride key+(r.paddedHeads key-1)*r.dimension+(r.width-1)+r.half
def tableLast (r : Request) := (r.sequence-1)*r.half+(r.width-1)

def apart (a b : Tensor) : Prop :=
  a.storage.elements=0 ∨ b.storage.elements=0 ∨
    a.storage.base+4*a.storage.elements≤b.storage.base ∨
    b.storage.base+4*b.storage.elements≤a.storage.base

/-- Actual prepared arguments, before the kernel. No preparation or returned
object execution is part of this boundary. -/
structure Call where
  geometry : Request
  q : Bool→Tensor
  tables : Bool→Tensor
  qLaws : ∀ k,Laws (q k)
  tableLaws : ∀ k,Laws (tables k)

def Call.source (c : Call) : ComputeKernel :=
  let r:=c.geometry
  Kernels.source .fp32 .fp32 "Q" "K" "C" "S"
    (r.stride Bool.false) (r.stride Bool.true) r.half r.half r.sequence r.batches
    r.qHeads r.kHeads r.dimension (r.paddedHeads Bool.false) (r.paddedHeads Bool.true)
    r.paddedDimension (max (r.paddedHeads Bool.false) (r.paddedHeads Bool.true)) Bool.false

/-- Existing finite domain and address checks at the actual JIT boundary. -/
structure Admitted (c : Call) : Prop where
  qShape : ∀ k,(c.q k).metadata.shape=[c.geometry.batches,c.geometry.sequence,c.geometry.heads k,c.geometry.dimension]
  tableShape : ∀ k,(c.tables k).metadata.shape=[c.geometry.sequence,c.geometry.half]
  qContiguous : ∀ k,(c.q k).metadata.isContiguous=Bool.true
  tableContiguous : ∀ k,(c.tables k).metadata.isContiguous=Bool.true
  qResolved : ∀ k,(c.q k).metadata.isNeg=Bool.false
  tableResolved : ∀ k,(c.tables k).metadata.isNeg=Bool.false
  qDevice : ∀ k,(c.q k).metadata.device.kind="cuda" ∧ (c.q k).metadata.device=(c.q Bool.false).metadata.device
  tableDevice : ∀ k,(c.tables k).metadata.device=(c.q Bool.false).metadata.device
  qApart : apart (c.q Bool.false) (c.q Bool.true)
  tableApart : ∀ k s,apart (c.q k) (c.tables s)
  coefficientsApart : apart (c.tables Bool.false) (c.tables Bool.true)
  alignedQ : ∀ k,0<(c.q k).metadata.numel →(c.q k).dataPtr.toNat%4=0
  alignedTable : ∀ k,0<(c.tables k).metadata.numel →(c.tables k).dataPtr.toNat%4=0
  metadataBounds : c.geometry.batches<2^31 ∧ c.geometry.sequence<2^31 ∧
    c.geometry.qHeads<2^31 ∧ c.geometry.kHeads<2^31 ∧ c.geometry.dimension<2^31 ∧
    c.geometry.paddedDimension<2^31 ∧ c.geometry.rows<2^31
  shapeBounds : ∀ k,c.geometry.paddedHeads k<2^31 ∧ c.geometry.stride k<2^31 ∧
    c.geometry.paddedHeads k*c.geometry.width≤1048576
  formedBounds : c.geometry.rows≠0 →∀ k,
    outputLast c.geometry k<2^31 ∧ tableLast c.geometry<2^31 ∧
    (c.q k).dataPtr.toNat+4*outputLast c.geometry k+4≤2^64 ∧
    (c.tables k).dataPtr.toNat+4*tableLast c.geometry+4≤2^64

theorem source_binding (c : Call) : c.source=kernel c.geometry := rfl

#axiomsClean source_binding
#specNonCircular Admitted avoiding [Call.source,FP.Structural.exec,Grid.run]
end
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
