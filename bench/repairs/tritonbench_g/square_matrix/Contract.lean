import VeriTile.Torch.Tensor
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural
import bench.repairs.tritonbench_g.square_matrix.Kernels

namespace VeriTile.Bench.Repairs.SquareMatrix
/-- The installed Triton helper's six shifts, including its zero case. -/
def block (n : Nat) : Nat :=
  if n=0 then 0 else
  let x := n-1
  let x := x ||| (x >>> 1)
  let x := x ||| (x >>> 2)
  let x := x ||| (x >>> 4)
  let x := x ||| (x >>> 8)
  let x := x ||| (x >>> 16)
  let x := x ||| (x >>> 32)
  x+1

def warps (B : Nat) : Nat :=
  let w := if 2048 ≤ B then 8 else 4
  if 4096 ≤ B then 16 else w

/-- Independent mathematical specification: elementwise square, not a matrix product. -/
def mathematical (value : ℝ) : ℝ := value * value

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def rows (x : TensorMetadata) : Nat := x.shape.headD 0
def cols (x : TensorMetadata) : Nat := x.shape[1]?.getD 0

structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  rank : x.shape.length = 2
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  extent : x.numel < 2^31
  /-- Natural alignment for FP32 memory accesses, on the actual observed
  pointer. Empty inputs launch nothing. See `documents/GpuMemoryModel.md`
  and its PTX addresses-as-operands reference. -/
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0
  frontend : 0 < x.numel → block (cols x) ≤ 2^20

/-- Bind the kernel's typed input cells to the original Tensor's contiguous
storage window. The shared Tensor laws connect the metadata to storage and
actual pointers; this predicate neither copies nor normalizes the input. -/
def InputCells (x : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws x)
    (h : InputContract x.metadata) (read : Nat → VeriTile.Torch.FP32.Word)
    (memory : VeriTile.Triton.RegionName → Nat →
      VeriTile.Triton.FP.Structural.Cell VeriTile.Torch.FP32.Word) : Prop :=
  ∀ i, i < x.metadata.numel → memory "x" i = .mk .real
    (read ((VeriTile.Torch.FP32.dataView x laws h.contiguous).address i))

inductive InputError where
  | tensor | layout | dtype | cuda | rank | contiguous | negative | extent | alignment | frontend
  deriving DecidableEq, Repr

def checkTensorInput (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if x.shape.length != 2 then .error .rank
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && x.dataPtr.toNat % 4 != 0 then .error .alignment
  else if x.numel > 0 && block (cols x) > 2^20 then .error .frontend
  else .ok ()

def checkInput : ArgumentMetadata → Except InputError Unit
  | .tensor x => checkTensorInput x
  | .nonTensor => .error .tensor

def AdmittedArgument : ArgumentMetadata → Prop
  | .tensor x => InputContract x
  | .nonTensor => False
end VeriTile.Bench.Repairs.SquareMatrix
