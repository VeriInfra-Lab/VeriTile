import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.f8_conversion_utils.Kernels
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Numerics

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Checked
open VeriTile.Torch

inductive FormatArgument where
  | e4 | e5 | other
  deriving DecidableEq

def inputDType (d : Direction) (x : TensorMetadata) : Prop :=
  match d with
  | .decode => x.dtype = .other "int8"
  | .encode => x.dtype = .other "float16" ∨ x.dtype = .floating .fp32

instance (d : Direction) (x : TensorMetadata) : Decidable (inputDType d x) := by
  cases d <;> unfold inputDType <;> infer_instance

def inputWidth (d : Direction) (x : TensorMetadata) : Nat :=
  match d with
  | .decode => 1
  | .encode => if x.dtype = .other "float16" then 2 else 4

def profile (d : Direction) (format : FormatArgument) (x : TensorMetadata) : Source.Profile :=
  match d,format with
  | .decode,.e4 => .decodeE4
  | .decode,_ => .decodeE5
  | .encode,.e4 => if x.dtype = .other "float16" then .encodeHalfE4 else .encodeSingleE4
  | .encode,_ => if x.dtype = .other "float16" then .encodeHalfE5 else .encodeSingleE5

structure InputContract (d : Direction) (format : FormatArgument) (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : inputDType d x
  format : format ≠ .other
  cuda : x.device.kind = "cuda"
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  alignment : 0 < x.numel → x.dataPtr.toNat % inputWidth d x = 0

inductive InputError where
  | tensor | layout | dtype | format | cuda | contiguous | negative | alignment
  deriving DecidableEq

def checkTensor (d : Direction) (format : FormatArgument) (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if ¬ inputDType d x then .error .dtype
  else if format = .other then .error .format
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if 0 < x.numel ∧ x.dataPtr.toNat % inputWidth d x ≠ 0 then .error .alignment
  else .ok ()

def checkInputs (d : Direction) (format : FormatArgument) : ArgumentMetadata → Except InputError Unit
  | .nonTensor => .error .tensor
  | .tensor x => checkTensor d format x

def AdmittedArguments (d : Direction) (format : FormatArgument) : ArgumentMetadata → Prop
  | .nonTensor => False
  | .tensor x => InputContract d format x

/-- Independent scalar bit formula. Decode extends finite FP8 exactly;
encode rounds to nearest even with finite saturation. Signed zeros,
infinities, subnormals, and class-preserving NaN policy remain in the domain.
This is a primitive semantic definition, not an optimization assumption. -/
noncomputable def conversionBits (policy : Numerics.Policy) (p : Source.Profile) :
    Bytes.Word p.inputWidth → Bytes.Word p.outputWidth := Numerics.reference policy p

/-- The mathematical cast-erasure formula, independently of the source. -/
def conversionReal (value : ℝ) : ℝ := value

/-- Ordinary Tensor/storage API facts. The storage offset is included in
data_ptr once. A contiguous Tensor's row-major logical index maps to the
corresponding physical element. No source result is part of this binding. -/
structure TensorStorage (width : Nat) (metadata : TensorMetadata) where
  storageBase : Nat
  storageOffset : Nat
  elements : Nat
  physicalIndex : Fin metadata.numel → Nat
  positive : 0 < width
  indexBounds : ∀ i, physicalIndex i < elements
  span : storageBase + elements*width ≤ 2^64
  pointer : metadata.dataPtr.toNat = storageBase + storageOffset*width
  contiguous : metadata.isContiguous = Bool.true →
    ∀ i, physicalIndex i = storageOffset + i.val

def TensorStorage.address (t : TensorStorage width metadata) (i : Fin metadata.numel) : Nat :=
  t.storageBase + t.physicalIndex i*width

/-- Logical input coordinates under the checked no-lazy-negation contract. -/
def logicalBits (t : TensorStorage width metadata) (_plain : metadata.isNeg = Bool.false) (heap : Bytes.Heap)
    (i : Fin metadata.numel) : Bytes.Word width := Bytes.read width heap (t.address i)

noncomputable def logicalReal (t : TensorStorage width metadata) (_plain : metadata.isNeg = Bool.false) (heap : Nat → ℝ)
    (i : Fin metadata.numel) : ℝ := heap (t.address i)

#specNonCircular conversionBits avoiding [Source.kernel]
end VeriTile.Bench.Repairs.F8ConversionUtils.Checked
