import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Kernels
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Contract
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
abbrev InputContract := Revision001.InputContract
abbrev checkInputs := Revision001.checkInputs
abbrev AdmittedArguments := Revision001.AdmittedArguments
abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev Laws := VeriTile.Torch.FP32.TensorLaws
abbrev Word := VeriTile.Torch.FP32.Word
abbrev rows := SharedContract.rows
abbrev columns := SharedContract.columns

/-- Actual kernel argument objects. Tensor API laws carry storage/contiguity
facts and no kernel result, successful allocation or host invocation. -/
structure Arguments where
  a : Tensor
  b : Tensor
  output : Tensor
  leftLaws : Laws a
  rightLaws : Laws b
  outputLaws : Laws output

def apart (x y : Tensor) : Prop :=
  x.storage.base+4*x.storage.elements ≤ y.storage.base ∨
    y.storage.base+4*y.storage.elements ≤ x.storage.base

/-- Per-kernel output descriptor supplied by the wrapper's explicit output
allocation. No allocator function, return-value or execution premise is used.
Both original inputs may be offset contiguous views and may overlap. -/
structure OutputContract (c : Arguments) : Prop where
  shape : c.output.metadata.shape=[rows c.a.metadata,columns c.b.metadata]
  contiguous : c.output.metadata.isContiguous=Bool.true
  offset : c.output.storageOffset=0
  noNeg : c.output.metadata.isNeg=Bool.false
  device : c.output.metadata.device=c.a.metadata.device
  alignment : 0 < c.output.metadata.numel → c.output.metadata.dataPtr.toNat%4=0
  apartLeft : apart c.output c.a
  apartRight : apart c.output c.b

structure KernelContract (c : Arguments) : Prop where
  inputs : InputContract c.a.metadata c.b.metadata
  output : OutputContract c

theorem check_inputs_iff (a b : VeriTile.Torch.ArgumentMetadata) :
    checkInputs a b=.ok () ↔ AdmittedArguments a b :=
  SharedContract.checkInputs_iff Bool.false Bool.true a b

#axiomsClean check_inputs_iff
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
