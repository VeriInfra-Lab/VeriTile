import VeriTile.Triton
import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton
structure ViewMetadata where
  recognized : Bool
  base : VeriTile.Torch.TensorMetadata
  viewDType : VeriTile.Torch.DType
  viewDevice : VeriTile.Torch.Device
  integerMetadata : Bool
  shape : List Nat
  strides : List Int
  region : RegionName
  storageBase : Nat
  storageBytes : Nat
  /-- Actual view.data_ptr() minus base.untyped_storage().data_ptr(). This
  signed API observation includes Tensor storage offsets and StridedBuffer's
  constructor offset; it is not an assumed address result. -/
  delta : Int

/-- Primitive observations are computed from shared PyTorch metadata. The
remaining fields describe a Tensor or StridedBuffer's actual signed view. -/
def ViewMetadata.strided (v : ViewMetadata) : Bool := v.base.isStrided
def ViewMetadata.fp32 (v : ViewMetadata) : Bool :=
  decide (v.viewDType = .floating .fp32 ∧ v.base.dtype = .floating .fp32)
def ViewMetadata.resolved (v : ViewMetadata) : Bool := !v.base.isNeg
def ViewMetadata.cuda (v : ViewMetadata) : Bool :=
  decide (v.base.device.kind = "cuda" ∧ v.viewDevice = v.base.device)
def ViewMetadata.device (v : ViewMetadata) : Nat := v.viewDevice.index

def count (v : ViewMetadata) := v.shape.headD 0
def stride (v : ViewMetadata) := v.strides.headD 0
def last (v : ViewMetadata) : Int := v.delta+4*((count v-1 : Nat):Int)*stride v
def lower (v : ViewMetadata) : Int := min v.delta (last v)
def upper (v : ViewMetadata) : Int := max v.delta (last v)
def beginByte (v : ViewMetadata) : Int := v.storageBase+lower v
def endByte (v : ViewMetadata) : Int := v.storageBase+upper v+4

structure ViewContract (v : ViewMetadata) : Prop where
  recognized : v.recognized=Bool.true
  strided : v.strided=Bool.true
  fp32 : v.fp32=Bool.true
  resolved : v.resolved=Bool.true
  cuda : v.cuda=Bool.true
  integerMetadata : v.integerMetadata=Bool.true
  rank : v.shape.length=1 ∧ v.strides.length=1
  strideWord : -(2^63:Int)≤stride v ∧ stride v<2^63
  aligned : v.storageBase%4=0 ∧ v.delta%4=0
  allocation : v.storageBase+v.storageBytes≤2^64
  bounds : 0<count v → 0≤lower v ∧ upper v+4≤v.storageBytes

structure InputContract (isFloat : Bool) (x y : ViewMetadata) : Prop where
  scalar : isFloat=Bool.true
  input : ViewContract x
  output : ViewContract y
  shape : count x=count y
  positive : 0<count x
  extent : count x<2^31
  device : x.device=y.device
  outputStride : stride y≠0
  separation : endByte x≤beginByte y ∨ endByte y≤beginByte x

inductive Error where
  | scalar | view | layout | dtype | negative | cuda | metadata | rank | stride | alignment | allocation | bounds
  | shape | empty | extent | device | outputStride | overlap
  deriving DecidableEq, Repr

def checkView (v : ViewMetadata) : Except Error Unit :=
  if !v.recognized then .error .view
  else if !v.strided then .error .layout
  else if !v.fp32 then .error .dtype
  else if !v.resolved then .error .negative
  else if !v.cuda then .error .cuda
  else if v.shape.length≠1 ∨ v.strides.length≠1 then .error .rank
  else if !v.integerMetadata then .error .metadata
  else if ¬(-(2^63:Int)≤stride v ∧ stride v<2^63) then .error .stride
  else if v.storageBase%4≠0 ∨ v.delta%4≠0 then .error .alignment
  else if v.storageBase+v.storageBytes>2^64 then .error .allocation
  else if 0<count v ∧ (lower v<0 ∨ v.storageBytes<upper v+4) then .error .bounds
  else .ok ()

def checkInputs (isFloat : Bool) (x y : ViewMetadata) : Except Error Unit :=
  if !isFloat then .error .scalar
  else match checkView x with
  | .error e => .error e
  | .ok () => match checkView y with
    | .error e => .error e
    | .ok () =>
      if count x≠count y then .error .shape
      else if count x=0 then .error .empty
      else if count x≥2^31 then .error .extent
      else if x.device≠y.device then .error .device
      else if stride y=0 then .error .outputStride
      else if ¬(endByte x≤beginByte y ∨ endByte y≤beginByte x) then .error .overlap
      else .ok ()

end VeriTile.Bench.Repairs.PowScalarTensor.Checked
