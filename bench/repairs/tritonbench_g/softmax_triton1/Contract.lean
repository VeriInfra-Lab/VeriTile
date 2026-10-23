import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.square_matrix.Contract

namespace VeriTile.Bench.Repairs.SoftmaxTriton1

/-- Empty inputs do not launch; one is a harmless proof width for that case. -/
def blockSize (n : Nat) : Nat := max 1 (VeriTile.Bench.Repairs.SquareMatrix.block n)

namespace Contiguous
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
abbrev block := VeriTile.Bench.Repairs.SquareMatrix.block

def rows (x : TensorMetadata) : Nat := x.shape.headD 0
def cols (x : TensorMetadata) : Nat := x.shape[1]?.getD 0

structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  rank : x.shape.length = 2
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  grid : 0 < x.numel → rows x < 2^31
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0
  frontend : 0 < x.numel → block (cols x) ≤ 2^20

inductive InputError where
  | tensor | layout | dtype | cuda | rank | contiguous | negative | grid | alignment | frontend
  deriving DecidableEq, Repr

def checkTensorInput (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if x.shape.length != 2 then .error .rank
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel > 0 && rows x >= 2^31 then .error .grid
  else if x.numel > 0 && x.dataPtr.toNat % 4 != 0 then .error .alignment
  else if x.numel > 0 && block (cols x) > 2^20 then .error .frontend
  else .ok ()

def checkInput : ArgumentMetadata → Except InputError Unit
  | .tensor x => checkTensorInput x
  | .nonTensor => .error .tensor

def AdmittedArgument : ArgumentMetadata → Prop
  | .tensor x => InputContract x
  | .nonTensor => False
end Contiguous
end VeriTile.Bench.Repairs.SoftmaxTriton1
