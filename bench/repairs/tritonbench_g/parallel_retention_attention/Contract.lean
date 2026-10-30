import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelStorage
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Checked

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
noncomputable section

/-- Kernel entry after the selected wrapper's original contiguous copies.
The wrapper establishes shape, dtype, device, tile and pointer guards; fresh
scratch allocations establish separation. No output or execution is assumed. -/
structure ForwardContract (s : Shape) (b : Bases) (tile : Tile) (output : Nat) : Prop where
  shape : ShapeContract s
  program : ProgramDomain s 128 tile
  fresh : Disjoint (forwardFootprint s tile output) (forwardInputCells s b)

structure BackwardContract (s : Shape) (b : Bases) (tile : Tile) (dq dk dv : Nat) : Prop where
  shape : ShapeContract s
  program : ProgramDomain s 64 tile
  separate : BackwardSeparate s tile dq dk dv
  fresh : Disjoint (keyFootprint s tile dq ∪ keyFootprint s tile dk ∪ valueFootprint s tile dv) (inputCells s b)

/-- Ordinary metadata of the actual kernel scratch Tensor, before the source
wrapper's sum(0). A shard dimension is retained, including partial features. -/
structure ScratchTensor (s : Shape) (kind : OutputKind) (base device : Nat) where
  metadata : VeriTile.Torch.TensorMetadata
  shape : metadata.shape=[kind.shards s,s.B,s.H,s.T,kind.width s]
  dtype : metadata.dtype=.floating .fp32
  layout : metadata.layout=.strided
  device : metadata.device=⟨"cuda",device⟩
  contiguous : metadata.isContiguous=Bool.true
  logical : metadata.isNeg=Bool.false
  pointer : metadata.dataPtr.toNat=4*base
  room : 4*(base+kind.count s)<2^64
  formed : PointerDomain (4*base)
    (if kind=.query ∨ kind=.key then Checked.formedKey s else Checked.formedValue s)

/-- Actual contiguous incoming cotangent at the backward kernel entry. -/
structure Cotangent (s : Shape) (b : Bases) (device : Nat) where
  metadata : VeriTile.Torch.TensorMetadata
  shape : metadata.shape=[s.B,s.H,s.T,s.DV]
  dtype : metadata.dtype=.floating .fp32
  layout : metadata.layout=.strided
  device : metadata.device=⟨"cuda",device⟩
  contiguous : metadata.isContiguous=Bool.true
  logical : metadata.isNeg=Bool.false
  pointer : metadata.dataPtr.toNat=4*b.g
  room : PointerDomain metadata.dataPtr.toNat (Checked.formedValue s)

/-- Metadata admission is exactly the existing checked wrapper predicate. -/
theorem admitted_shape (s : Shape) (observed : Checked.Inputs) (admitted : Checked.admit s observed=Bool.true) :
    ShapeContract s := ((Checked.admit_iff s observed).mp admitted).geometry

#axiomsClean admitted_shape
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
