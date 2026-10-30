import bench.tritonbench_g.rms_matmul_rbe.Proofs.Contract

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Contract
open VeriTile.Bench.Repairs.RmsMatmulRbe.Contract

structure Inputs where
  (x weight rms : Tensor)
  scalars : Scalars
  (rotaryBool rotary : Bool)
def Inputs.reads (i : Inputs) := [i.x,i.weight,i.rms]
def rmsTypes : List VeriTile.Torch.DType := [fp32,fp16,int8]

/-- The single-projection source fixes epsilon/theta and retains its original
RMS Tensor.to(weight.dtype) conversion. -/
noncomputable def effectiveScalars (i : Inputs) : Scalars :=
  {i.scalars with
    epsilonNumeric := Bool.true
    epsilonFinite := Bool.true
    epsilon := 1/1000000
    thetaNumeric := Bool.true
    thetaFinite := Bool.true
    theta := 10000}
def InputContract (i : Inputs) : Prop :=
  TensorAdmitted i.x [fp32] ∧ TensorAdmitted i.weight weightTypes ∧ TensorAdmitted i.rms rmsTypes ∧
  i.weight.metadata.device=i.x.metadata.device ∧ i.rms.metadata.device=i.x.metadata.device ∧
  i.rotaryBool=Bool.true ∧ i.x.metadata.shape.length=3 ∧ i.weight.metadata.shape.length=2 ∧
  ScalarAdmitted (effectiveScalars i) (dim i.x 0) (dim i.x 1) i.rotary ∧
  0<dim i.x 2 ∧ dim i.weight 1=dim i.x 2 ∧ i.rms.metadata.shape=[dim i.x 2] ∧
  dim i.weight 0=i.scalars.heads.toNat*i.scalars.dim.toNat ∧ SameReadFormat i.reads

noncomputable def checkInputs (i : Inputs) : Except Error Unit :=
  if !tensorGuard i.x [fp32] || !tensorGuard i.weight weightTypes || !tensorGuard i.rms rmsTypes then .error .tensor
  else if i.weight.metadata.device≠i.x.metadata.device ∨ i.rms.metadata.device≠i.x.metadata.device then .error .device
  else if !i.rotaryBool then .error .scalar
  else if i.x.metadata.shape.length≠3 ∨ i.weight.metadata.shape.length≠2 then .error .shape
  else if !scalarGuard (effectiveScalars i) (dim i.x 0) (dim i.x 1) i.rotary then .error .scalar
  else if dim i.x 2=0 ∨ dim i.weight 1≠dim i.x 2 ∨ i.rms.metadata.shape≠[dim i.x 2] ∨
    dim i.weight 0≠i.scalars.heads.toNat*i.scalars.dim.toNat then .error .shape
  else if !pythonReadFormatGuard i.reads then .error .alias
  else .ok ()

end VeriTile.Bench.Repairs.RmsRbeMatmul.Contract
