import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.FiniteHost
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.InputAdapter
open VeriTile.Triton FP.Structural
open VeriTile.Torch

open SharedContract TensorBinding
noncomputable section
set_option maxHeartbeats 3000000

def tensor (t : TensorBinding.Tensor) (region : RegionName) : Host.Tensor where
  objectId := t.object
  device := t.metadata.device.index
  rank := t.metadata.shape.length
  rows := rows t.metadata
  columns := columns t.metadata
  rowStride := columns t.metadata
  columnStride := 1
  capacity := t.metadata.numel
  region := region
  fp32 := t.metadata.isFP32
  contiguous := t.metadata.isContiguous
  negative := t.metadata.isNeg
  cuda := t.metadata.isCuda

def arguments (a b : TensorBinding.Tensor) (ra rb : RegionName) (default : Bool) : Host.Arguments :=
  ⟨tensor a ra,tensor b rb,Host.sourceConfig,default⟩

theorem dense (t : TensorBinding.Tensor) (r : RegionName) (rank : t.metadata.shape.length=2) :
    Host.Dense (tensor t r) := by
  simp [Host.Dense,Core.DenseView,tensor,matrix_numel t.metadata rank]

theorem valid (a b : TensorBinding.Tensor) (ra rb : RegionName) (default : Bool)
    (h : SharedContract.InputContract Bool.true default a.metadata b.metadata) :
    Host.Valid (arguments a b ra rb default) := by
  have rightDevice : b.metadata.device.index=a.metadata.device.index :=
    congrArg Device.index h.device.symm
  constructor
  · simp only [Host.Guard,arguments,tensor,TensorMetadata.isFP32,TensorMetadata.isCuda]
    simp [h.rank.1,h.rank.2,h.dtype.1,h.dtype.2,h.cuda.1,h.cuda.2,h.fixedDevice rfl,
      rightDevice,h.defaultDtype rfl,h.contiguous.1,h.contiguous.2,h.noNeg.1,h.noNeg.2,h.dimensions]
  · rfl
  · exact dense a ra h.rank.1
  · exact dense b rb h.rank.2
  all_goals norm_num [arguments,Host.sourceConfig]

/-- Local successful empty-allocation facts. They constrain the fresh
allocation's metadata/capacity, never any output value. -/
structure OutputABI (c : Host.Tensor) : Prop where
  rowStride : c.rowStride=c.columns
  columnStride : c.columnStride=1
  bytes : c.capacity*4 ≤ 2^64

theorem canonical {α : Type} (a b : TensorBinding.Tensor) (ra rb : RegionName) (default : Bool)
    (initial allocated : State α) (c : Host.Tensor)
    (allocation : Host.Allocated (arguments a b ra rb default) initial allocated c)
    (abi : OutputABI c) : FiniteSource.Canonical (Host.parameters (arguments a b ra rb default) c) := by
  constructor <;> try rfl
  · exact abi.rowStride.trans allocation.columns
  · exact abi.columnStride

theorem storage {α : Type} (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b)
    (ra rb : RegionName) (default : Bool)
    (h : SharedContract.InputContract Bool.true default a.metadata b.metadata)
    (initial allocated : State α) (c : Host.Tensor)
    (allocation : Host.Allocated (arguments a b ra rb default) initial allocated c)
    (abi : OutputABI c) : FiniteIndex.StorageBounds (Host.parameters (arguments a b ra rb default) c) := by
  have left := TensorBinding.span a la h.contiguous.1
  have right := TensorBinding.span b lb h.contiguous.2
  rw [matrix_numel a.metadata h.rank.1] at left
  rw [matrix_numel b.metadata h.rank.2,←h.dimensions] at right
  constructor
  · exact left
  · exact right
  · have cap := allocation.dense.1
    rw [allocation.rows,allocation.columns] at cap
    have bytes := abi.bytes
    change rows a.metadata*columns b.metadata ≤ c.capacity at cap
    change rows a.metadata*columns b.metadata*4 ≤ 2^64
    omega

theorem ready (a b : TensorBinding.Tensor) (ra rb : RegionName) (default : Bool)
    (h : SharedContract.InputContract Bool.true default a.metadata b.metadata) (c : Host.Tensor)
    (launch : FiniteIndex.count (Host.parameters (arguments a b ra rb default) c) < 2^31) :
    FiniteIndex.Ready (Host.parameters (arguments a b ra rb default) c) := by
  constructor
  · exact h.extent.1
  · exact h.extent.2.2
  · exact h.extent.2.1
  all_goals first | exact launch | norm_num [Host.parameters,arguments,Host.sourceConfig]

#axiomsClean valid
#axiomsClean canonical
#axiomsClean storage
#axiomsClean ready
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.InputAdapter
