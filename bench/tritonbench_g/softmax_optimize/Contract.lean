import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.SoftmaxOptimize
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def InputContract (x : TensorMetadata) : Prop :=
  x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
  x.shape.length=2 ∧ x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧
  x.numel<2^31 ∧ (0<x.numel → x.dataPtr.toNat%4=0)

inductive InputError where | tensor | metadata | rank | extent | alignment
  deriving DecidableEq, Repr

def check (arg : ArgumentMetadata) : Except InputError Unit :=
  match arg with
  | .nonTensor => .error .tensor
  | .tensor x =>
    if x.layout≠.strided ∨ x.dtype≠.floating .fp32 ∨ x.device.kind≠"cuda" then .error .metadata
    else if x.shape.length≠2 then .error .rank
    else if x.isContiguous≠Bool.true ∨ x.isNeg≠Bool.false then .error .metadata
    else if x.numel≥2^31 then .error .extent
    else if x.numel≠0 ∧ x.dataPtr.toNat%4≠0 then .error .alignment
    else .ok ()

def rows (x : TensorMetadata) : Nat := x.shape[0]?.getD 0
def columns (x : TensorMetadata) : Nat := x.shape[1]?.getD 0

/-- Ordinary successful empty_like result and fresh allocation. Alignment and
span validity are needed only for nonempty results. -/
def Allocated (x y : TensorMetadata) : Prop :=
  y.shape=x.shape ∧ y.device=x.device ∧ y.dtype=.floating .fp32 ∧
  y.layout=.strided ∧ y.isContiguous=Bool.true ∧ y.isNeg=Bool.false ∧
  (0<y.numel → y.dataPtr.toNat%4=0 ∧ y.dataPtr.toNat+4*y.numel≤2^64) ∧
  (x.dataPtr.toNat+4*x.numel≤y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤x.dataPtr.toNat)
end VeriTile.Bench.Repairs.SoftmaxOptimize
