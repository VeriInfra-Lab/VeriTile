import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.Int8Quantization.Direct
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
def batchShape (x : TensorMetadata) : List Nat := x.shape.take (x.shape.length-2)
def batches (x : TensorMetadata) : Nat := (batchShape x).prod
def length (x : TensorMetadata) : Nat := (x.shape.drop (x.shape.length-2))[0]?.getD 0
def channels (x : TensorMetadata) : Nat := (x.shape.drop (x.shape.length-2))[1]?.getD 0
def blocks (x : TensorMetadata) (block : Nat) : Nat := (length x+block-1)/block
def scaleCount (x : TensorMetadata) (block : Nat) : Nat := batches x*blocks x block
def scaleShape (x : TensorMetadata) (block : Nat) : List Nat := batchShape x++[blocks x block,1]
structure InputContract (x : TensorMetadata) (block : Nat) : Prop where
  rank : x.shape.length=3 ∨ x.shape.length=4
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
  contiguous : x.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false
  blockPositive : 0< block
  tile : block*channels x≤2^20
  inputExtent : x.numel<2^61
  scaleExtent : scaleCount x block<2^61
  alignment : 0< x.numel → x.dataPtr.toNat%4=0
inductive InputError where
  | tensor | rank | dtype | cuda | contiguous | block | tile | extent | alignment
  deriving DecidableEq,Repr
inductive BlockArgument where
  | nonInteger | integer (value : Int)
  deriving DecidableEq,Repr

def checkTensorInputs (x : TensorMetadata) (block : Nat) : Except InputError Unit :=
  if x.shape.length != 3 && x.shape.length != 4 then .error .rank
  else if x.layout != .strided || x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous || x.isNeg then .error .contiguous
  else if block=0 then .error .block
  else if block*channels x>2^20 then .error .tile
  else if x.numel≥2^61 || scaleCount x block≥2^61 then .error .extent
  else if x.numel>0 && x.dataPtr.toNat%4 != 0 then .error .alignment
  else .ok ()
def checkInputs (x : ArgumentMetadata) (block : BlockArgument) : Except InputError Unit :=
  match x,block with
  | .tensor t,.integer b => if b≤0 then .error .block else checkTensorInputs t b.toNat
  | .nonTensor,_ => .error .tensor
  | _,.nonInteger => .error .block
def AdmittedArguments (x : ArgumentMetadata) (block : BlockArgument) : Prop :=
  match x,block with
  | .tensor t,.integer b => 0< b ∧ InputContract t b.toNat
  | _,_ => False
end VeriTile.Bench.Repairs.Int8Quantization.Direct
