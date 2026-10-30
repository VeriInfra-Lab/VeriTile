import VeriTile.Torch.Tensor
import bench.tritonbench_g.softmax_triton3.Kernels
import bench.tritonbench_g.square_matrix.Contract

namespace VeriTile.Bench.Repairs.SoftmaxTriton3
def blockSize (n : Nat) : Nat := max (VeriTile.Bench.Repairs.SquareMatrix.block n) 2
def numWarps (block : Nat) : Nat := if block ≥ 4096 then 16 else if block ≥ 2048 then 8 else 4
end VeriTile.Bench.Repairs.SoftmaxTriton3

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def rows (x : TensorMetadata) := x.shape.dropLast.prod
def cols (x : TensorMetadata) := x.shape.getLastD 0

structure TensorContract (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  rank : x.shape ≠ []
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  grid : 0 < x.numel → rows x < 2^31
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0
  frontend : 0 < x.numel → blockSize (cols x) ≤ 2^20

structure InputContract (x : TensorMetadata) (mask : Option TensorMetadata) (dim : Int) : Prop where
  input : TensorContract x
  dimension : dim = -1 ∨ dim = (x.shape.length : Int)-1
  mask : ∀ m,mask=some m → TensorContract m ∧ m.device=x.device ∧ cols m=cols x ∧ rows m=rows x

inductive InputError where
  | tensor | layout | dtype | cuda | rank | contiguous | negative | grid | alignment | frontend
  | dimension | device | width | rows
  deriving DecidableEq, Repr

def checkTensor (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if x.shape == [] then .error .rank
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel > 0 && rows x >= 2^31 then .error .grid
  else if x.numel > 0 && x.dataPtr.toNat % 4 != 0 then .error .alignment
  else if x.numel > 0 && blockSize (cols x) > 2^20 then .error .frontend
  else .ok ()

def checkTyped (x : TensorMetadata) (mask : Option TensorMetadata) (dim : Int) : Except InputError Unit :=
  match checkTensor x with
  | .error e => .error e
  | .ok () =>
    if dim != -1 && dim != (x.shape.length : Int)-1 then .error .dimension
    else match mask with
    | none => .ok ()
    | some m =>
      match checkTensor m with
      | .error e => .error e
      | .ok () =>
        if m.device != x.device then .error .device
        else if cols m != cols x then .error .width
        else if rows m != rows x then .error .rows
        else .ok ()

def checkInput (x : ArgumentMetadata) (mask : Option ArgumentMetadata) (dim : Int) : Except InputError Unit :=
  match x,mask with
  | .tensor x,none => checkTyped x none dim
  | .tensor x,some (.tensor m) => checkTyped x (some m) dim
  | _,_ => .error .tensor

def AdmittedArguments (x : ArgumentMetadata) (mask : Option ArgumentMetadata) (dim : Int) : Prop :=
  match x,mask with
  | .tensor x,none => InputContract x none dim
  | .tensor x,some (.tensor m) => InputContract x (some m) dim
  | _,_ => False
end VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
