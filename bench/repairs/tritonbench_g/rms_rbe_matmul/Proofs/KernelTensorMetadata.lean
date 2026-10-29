import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelTensorMetadata
open Contract (Inputs)

/-- Ordinary output view metadata. This boundary states no numerical value,
source execution, compiler lowering, reduction or whole-output relation. -/
theorem returned (i : Inputs) (a : Checked.Allocation i) (returns : Checked.Returns i a) :
    returns.tensor.metadata.shape=
      [(Checked.shape i).B,(Checked.shape i).M,(Checked.shape i).heads,(Checked.shape i).headDim] ∧
    returns.tensor.metadata.dtype=i.weight.metadata.dtype ∧
    returns.tensor.metadata.device=i.x.metadata.device ∧
    returns.tensor.metadata.dataPtr=a.out.metadata.dataPtr ∧
    returns.tensor.base=a.out.base ∧ returns.tensor.offset=a.out.offset := by
  exact ⟨returns.view.shape,returns.view.dtype.trans a.outDType,
    returns.view.device.trans a.outDevice,returns.view.pointer,returns.view.base,returns.view.offset⟩

#axiomsClean returned
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelTensorMetadata
