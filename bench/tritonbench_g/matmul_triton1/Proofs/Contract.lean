import VeriTile.Torch.Tensor

/-! The two concrete matrix wrappers share a contiguous FP32 policy.
`fixedCuda0` records matmul1's original fixed output device/default dtype;
matmul2 allocates on the input device with an explicit FP32 dtype. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract
open VeriTile.Torch

def rows (t : TensorMetadata) : Nat := t.shape[0]?.getD 0
def columns (t : TensorMetadata) : Nat := t.shape[1]?.getD 0

structure InputContract (fixedCuda0 defaultFP32 : Bool) (x y : TensorMetadata) : Prop where
  layout : x.layout = .strided ∧ y.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32
  cuda : x.device.kind = "cuda" ∧ y.device.kind = "cuda"
  device : x.device = y.device
  fixedDevice : fixedCuda0 = Bool.true → x.device.index = 0
  defaultDtype : fixedCuda0 = Bool.true → defaultFP32 = Bool.true
  rank : x.shape.length = 2 ∧ y.shape.length = 2
  dimensions : columns x = rows y
  contiguous : x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false ∧ y.isNeg = Bool.false
  extent : rows x < 2^63 ∧ columns x < 2^63 ∧ columns y < 2^63
  alignment : (0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧
    (0 < y.numel → y.dataPtr.toNat % 4 = 0)

inductive InputError where
  | tensors | layout | dtype | cuda | device | fixedDevice | defaultDtype
  | rank | dimensions | contiguous | negative | extent | alignment
  deriving DecidableEq, Repr

def checkTensorInputs (fixedCuda0 defaultFP32 : Bool) (x y : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || y.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || y.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && y.isCuda) then .error .cuda
  else if x.device != y.device then .error .device
  else if fixedCuda0 && x.device.index != 0 then .error .fixedDevice
  else if fixedCuda0 && !defaultFP32 then .error .defaultDtype
  else if x.shape.length != 2 || y.shape.length != 2 then .error .rank
  else if columns x != rows y then .error .dimensions
  else if !(x.isContiguous && y.isContiguous) then .error .contiguous
  else if x.isNeg || y.isNeg then .error .negative
  else if rows x >= 2^63 || columns x >= 2^63 || columns y >= 2^63 then .error .extent
  else if (x.numel > 0 && x.dataPtr.toNat % 4 != 0) ||
      (y.numel > 0 && y.dataPtr.toNat % 4 != 0) then .error .alignment
  else .ok ()

def checkInputs (fixedCuda0 defaultFP32 : Bool) (x y : ArgumentMetadata) : Except InputError Unit :=
  match x,y with
  | .tensor x,.tensor y => checkTensorInputs fixedCuda0 defaultFP32 x y
  | _,_ => .error .tensors

def AdmittedArguments (fixedCuda0 defaultFP32 : Bool) (x y : ArgumentMetadata) : Prop :=
  match x,y with
  | .tensor x,.tensor y => InputContract fixedCuda0 defaultFP32 x y
  | _,_ => False
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001
abbrev InputContract := SharedContract.InputContract Bool.true
abbrev checkInputs := SharedContract.checkInputs Bool.true
abbrev AdmittedArguments := SharedContract.AdmittedArguments Bool.true
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001
