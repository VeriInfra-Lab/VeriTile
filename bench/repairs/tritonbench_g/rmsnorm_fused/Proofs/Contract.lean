import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Admission
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Host

namespace VeriTile.Bench.Repairs.RmsnormFused
open VeriTile.Torch
noncomputable section
def request (x weight : TensorInfo) (eps : EpsilonInfo) (contract : InputContract x weight eps) : Host.Request where
  geometry := geometry x
  valid := ⟨contract.widthPositive,contract.block,block_bound _,by
    rw [geometry_size x contract.rank]
    exact contract.byteExtent⟩
  gridFits := by rw [geometry_rows x contract.rank contract.widthPositive]; exact contract.grid
  epsilon := eps.value
  positiveEpsilon := contract.epsilonPositive

/-- Same-call metadata observation. Only runtime tensor facts occur here. -/
structure Observes (x : TensorInfo) (t : Host.Tensor) : Prop where
  shape : t.shape=x.shape
  pointer : t.dataPtr=x.dataPtr.toNat
  dtype : t.dtype="float32"
  device : t.device=x.device.index
  cuda : t.cuda=x.isCuda
  layout : t.layout="strided"
  contiguous : t.contiguous=x.isContiguous
  offset : t.storageOffset=x.storageOffset
  negative : t.lazyNeg=x.isNeg
  conjugate : t.lazyConj=x.isConj

#axiomsClean checkInput_iff
#axiomsClean geometry_size
#axiomsClean geometry_rows
end
end VeriTile.Bench.Repairs.RmsnormFused
