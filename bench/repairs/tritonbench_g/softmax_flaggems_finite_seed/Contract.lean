import VeriTile.Torch.Tensor
import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridMetadata

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

def TensorContract (x : TensorMetadata) : Prop :=
  x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
  x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧ x.numel<2^31 ∧
  (0<x.numel → x.dataPtr.toNat%4=0)

def ForwardContract (x : TensorMetadata) (dim : Int) (dtype : Option VeriTile.Torch.DType) : Prop :=
  TensorContract x ∧ -(x.shape.length : Int)≤dim ∧ dim<(x.shape.length : Int) ∧
  (dtype=none ∨ dtype=some (.floating .fp32))

def GradientContract (x grad : TensorMetadata) : Prop :=
  TensorContract grad ∧ grad.shape=x.shape ∧ grad.device=x.device

inductive InputError where | tensor | metadata | extent | alignment | dimension | dtype | gradient
  deriving DecidableEq, Repr

def checkTensor (x : TensorMetadata) : Except InputError Unit :=
  if x.layout≠.strided ∨ x.dtype≠.floating .fp32 ∨ x.device.kind≠"cuda" then .error .metadata
  else if x.isContiguous≠Bool.true ∨ x.isNeg≠Bool.false then .error .metadata
  else if x.numel≥2^31 then .error .extent
  else if x.numel≠0 ∧ x.dataPtr.toNat%4≠0 then .error .alignment
  else .ok ()

def checkForward (x : ArgumentMetadata) (dim : Int) (dtype : Option VeriTile.Torch.DType) : Except InputError Unit :=
  match x with
  | .nonTensor => .error .tensor
  | .tensor x => match checkTensor x with
    | .error e => .error e
    | .ok () =>
      if ¬(-(x.shape.length : Int)≤dim ∧ dim<(x.shape.length : Int)) then .error .dimension
      else if ¬(dtype=none ∨ dtype=some (.floating .fp32)) then .error .dtype
      else .ok ()

def checkGradient (x : TensorMetadata) (g : ArgumentMetadata) : Except InputError Unit :=
  match g with
  | .nonTensor => .error .tensor
  | .tensor g => match checkTensor g with
    | .error e => .error e
    | .ok () => if g.shape≠x.shape ∨ g.device≠x.device then .error .gradient else .ok ()

/-- Python's strict `type(dim) is int`; booleans and other objects are rejected. -/
inductive DimArgument where
  | integer (value : Int)
  | other
  deriving DecidableEq

def checkDynamic (x : ArgumentMetadata) (dim : DimArgument) (dtype : Option VeriTile.Torch.DType) : Except InputError Unit :=
  match dim with
  | .integer d => checkForward x d dtype
  | .other => .error .dimension

def DynamicContract (x : ArgumentMetadata) (dim : DimArgument) (dtype : Option VeriTile.Torch.DType) : Prop :=
  ∃ t d, x=.tensor t ∧ dim=.integer d ∧ ForwardContract t d dtype

end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed
