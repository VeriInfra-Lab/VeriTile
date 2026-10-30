import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.MatrixVectorMultip
open VeriTile.Torch

def rows (x : TensorMetadata) : Nat := x.shape[0]?.getD 0
def columns (x : TensorMetadata) : Nat := x.shape[1]?.getD 0
def vectorLength (y : TensorMetadata) : Nat := y.shape[0]?.getD 0

structure InputContract (x y : TensorMetadata) : Prop where
  layout : x.layout = .strided ∧ y.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32
  cuda : x.device.kind = "cuda" ∧ y.device.kind = "cuda"
  device : x.device = y.device
  rank : x.shape.length = 2 ∧ y.shape.length = 1
  dimensions : columns x = vectorLength y
  contiguous : x.isContiguous = true ∧ y.isContiguous = true
  noNeg : x.isNeg = false ∧ y.isNeg = false
  extent : rows x < 2^63 ∧ columns x < 2^63
  alignment : (0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧
    (0 < y.numel → y.dataPtr.toNat % 4 = 0)

inductive InputError where
  | tensors | layout | dtype | cuda | device | rank | dimensions
  | contiguous | negative | extent | alignment
  deriving DecidableEq, Repr

def checkTensorInputs (x y : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || y.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || y.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && y.isCuda) then .error .cuda
  else if x.device != y.device then .error .device
  else if x.shape.length != 2 || y.shape.length != 1 then .error .rank
  else if columns x != vectorLength y then .error .dimensions
  else if !(x.isContiguous && y.isContiguous) then .error .contiguous
  else if x.isNeg || y.isNeg then .error .negative
  else if rows x >= 2^63 || columns x >= 2^63 then .error .extent
  else if (x.numel > 0 && x.dataPtr.toNat % 4 != 0) ||
      (y.numel > 0 && y.dataPtr.toNat % 4 != 0) then .error .alignment
  else .ok ()

def checkInputs (x y : ArgumentMetadata) : Except InputError Unit :=
  match x, y with
  | .tensor x, .tensor y => checkTensorInputs x y
  | _, _ => .error .tensors

def AdmittedArguments (x y : ArgumentMetadata) : Prop :=
  match x, y with
  | .tensor x, .tensor y => InputContract x y
  | _, _ => False

/-- All 48 original autotune configurations; stages and warps affect lowering,
but do not change the mathematical source coordinates. -/
structure KernelConfig where
  blockRows : Nat
  blockColumns : Nat
  stages : Nat
  warps : Nat
  rows_mem : blockRows ∈ [1, 2, 4, 8]
  columns_mem : blockColumns ∈ [32, 64, 128]
  stages_mem : stages ∈ [3, 4]
  warps_mem : warps ∈ [4, 8]

/-- Flat-memory placement supplied for one kernel invocation. Bases are FP32
word addresses, so accessed words are aligned. Input regions may overlap;
output bytes are separate from each active input word. These are backing-memory
facts, not extra Python guards or an allocator/launch proof. -/
structure KernelMemory (N M : Nat) where
  matrixBase : Nat
  vectorBase : Nat
  outputBase : Nat
  matrixBound : ∀ i j, i < N → j < M →
    4 * (matrixBase + i * M + j) + 4 ≤ 2^64
  vectorBound : ∀ j, j < M → 4 * (vectorBase + j) + 4 ≤ 2^64
  outputBound : ∀ i, i < N → 4 * (outputBase + i) + 4 ≤ 2^64
  matrixSeparate : ∀ i r j, i < N → r < N → j < M →
    outputBase + i ≠ matrixBase + r * M + j
  vectorSeparate : ∀ i j, i < N → j < M → outputBase + i ≠ vectorBase + j

/-- Actual checked input pointers and ordinary legal program-id conditions
bind the source's natural coordinates to its uint64 execution model. Empty
inputs require no pointer binding. No output-value premise occurs here. -/
structure KernelContract (x y : TensorMetadata) (c : KernelConfig) (pid : Nat)
    (memory : KernelMemory (rows x) (columns x)) : Prop where
  inputs : InputContract x y
  matrixPointer : 0 < x.numel → x.dataPtr.toNat = 4 * memory.matrixBase
  vectorPointer : 0 < y.numel → y.dataPtr.toNat = 4 * memory.vectorBase
  program : pid < (rows x + c.blockRows - 1) / c.blockRows
  programABI : pid < 2^31

end VeriTile.Bench.Repairs.MatrixVectorMultip
