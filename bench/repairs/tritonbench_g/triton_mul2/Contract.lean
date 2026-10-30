import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.TritonMul2
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def supportedBlocks : List Nat := [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576]

structure InputContract (x : TensorMetadata) (block : Nat) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  extent : x.numel < 2^31
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0
  block : block ∈ supportedBlocks

inductive InputError where
  | tensor | layout | dtype | cuda | contiguous | negative | extent | alignment | block
  deriving DecidableEq, Repr

/-- Python bool and non-integer objects are classified as nonInteger. -/
inductive BlockArgument where
  | nonInteger | integer (value : Int)
  deriving DecidableEq, Repr

def checkTensorInputs (x : TensorMetadata) (block : Nat) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && x.dataPtr.toNat % 4 != 0 then .error .alignment
  else if block ∉ supportedBlocks then .error .block
  else .ok ()

/-- Dynamic admission predicate; no attributes are fabricated for non-Tensors.
Rejection order for malformed block objects is immaterial to successful admission. -/
def checkInputs (x : ArgumentMetadata) (block : BlockArgument) : Except InputError Unit :=
  match x, block with
  | .nonTensor, _ => .error .tensor
  | _, .nonInteger => .error .block
  | .tensor t, .integer b =>
      if b<0 then .error .block else checkTensorInputs t b.toNat

def AdmittedArguments (x : ArgumentMetadata) (block : BlockArgument) : Prop :=
  match x, block with
  | .tensor t, .integer b => 0≤b ∧ InputContract t b.toNat
  | _, _ => False
/-- Independent mathematical target, separate from kernel execution. -/
noncomputable def mathematical (n : Nat) (initial final : Nat → ℝ) : Prop :=
  ∀ i, i<n → final i = 2*initial i

end VeriTile.Bench.Repairs.TritonMul2
