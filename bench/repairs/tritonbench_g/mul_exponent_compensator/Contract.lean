import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
def supportedBlocks : List Nat := (List.range 21).map (fun k => 2^k)

inductive BlockArgument where
  | nonInteger
  | integer (value : Int)
  deriving DecidableEq, Repr

structure InputContract (x : TensorMetadata) (current B : Nat) : Prop where
  rank : x.shape ≠ []
  block : B ∈ supportedBlocks
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  grid : (x.numel+B-1)/B ≤ 2^31-1
  device : 0<x.numel → x.device = ⟨"cuda",current⟩
  alignment : 0<x.numel → x.dataPtr.toNat%4=0

inductive InputError where
  | tensor | rank | blockType | blockValue | layout | dtype | cuda | contiguous | negative | grid | device | alignment
  deriving DecidableEq, Repr

def checkTensorInputs (x : TensorMetadata) (current B : Nat) : Except InputError Unit :=
  if x.shape = [] then .error .rank
  else if B ∉ supportedBlocks then .error .blockValue
  else if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if (x.numel+B-1)/B > 2^31-1 then .error .grid
  else if x.numel > 0 && x.device != ⟨"cuda",current⟩ then .error .device
  else if x.numel > 0 && x.dataPtr.toNat%4 != 0 then .error .alignment
  else .ok ()

/-- Python bool uses its ordinary integer subclass value (False=0, True=1). -/
def checkInputs (x : ArgumentMetadata) (current : Nat) (block : BlockArgument) : Except InputError Unit :=
  match x with
  | .nonTensor => .error .tensor
  | .tensor x =>
    if x.shape=[] then .error .rank
    else match block with
      | .nonInteger => .error .blockType
      | .integer b => if b<0 then .error .blockValue else checkTensorInputs x current b.toNat

def AdmittedArguments (x : ArgumentMetadata) (current : Nat) (block : BlockArgument) : Prop :=
  match x,block with
  | .tensor x,.integer b => 0≤b ∧ InputContract x current b.toNat
  | _,_ => False
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
