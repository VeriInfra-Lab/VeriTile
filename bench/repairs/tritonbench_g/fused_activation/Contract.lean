import VeriTile.Torch.Tensor
import Init.Data.Nat.Power2.Basic

namespace VeriTile.Bench.Repairs.FusedActivation
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def block (n : Nat) : Nat := min 2048 n.nextPowerOfTwo
def separate (a na b nb : Nat) : Prop := a+4*na ≤ b ∨ b+4*nb ≤ a
instance (a na b nb : Nat) : Decidable (separate a na b nb) :=
  inferInstanceAs (Decidable (_ ∨ _))

def InputContract (x bias input : TensorMetadata) : Prop :=
  (x.layout = .strided ∧ bias.layout = .strided ∧ input.layout = .strided) ∧
  (x.dtype = .floating .fp32 ∧ bias.dtype = .floating .fp32 ∧ input.dtype = .floating .fp32) ∧
  (x.device.kind = "cuda" ∧ bias.device.kind = "cuda" ∧ input.device.kind = "cuda") ∧
  (x.device = bias.device ∧ x.device = input.device) ∧
  (x.isContiguous = Bool.true ∧ bias.isContiguous = Bool.true ∧ input.isContiguous = Bool.true) ∧
  (x.isNeg = Bool.false ∧ bias.isNeg = Bool.false ∧ input.isNeg = Bool.false) ∧
  x.numel ≤ input.numel ∧ x.numel < 2^31 ∧ bias.numel < 2^31 ∧
  (0 < x.numel → 0 < bias.numel ∧
    (x.dataPtr.toNat%4=0 ∧ bias.dataPtr.toNat%4=0 ∧ input.dataPtr.toNat%4=0) ∧
    separate x.dataPtr.toNat x.numel bias.dataPtr.toNat bias.numel ∧
    separate x.dataPtr.toNat x.numel input.dataPtr.toNat x.numel)

inductive InputError where
  | tensors | layout | dtype | cuda | device | contiguous | negative | inputExtent
  | extent | bias | alignment | alias
  deriving DecidableEq, Repr

def checkTensorInputs (x b i : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || b.layout != .strided || i.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || b.dtype != .floating .fp32 || i.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && b.isCuda && i.isCuda) then .error .cuda
  else if x.device != b.device || x.device != i.device then .error .device
  else if !(x.isContiguous && b.isContiguous && i.isContiguous) then .error .contiguous
  else if x.isNeg || b.isNeg || i.isNeg then .error .negative
  else if i.numel < x.numel then .error .inputExtent
  else if x.numel >= 2^31 || b.numel >= 2^31 then .error .extent
  else if x.numel = 0 then .ok ()
  else if b.numel = 0 then .error .bias
  else if x.dataPtr.toNat%4 != 0 || b.dataPtr.toNat%4 != 0 || i.dataPtr.toNat%4 != 0 then .error .alignment
  else if !(decide (separate x.dataPtr.toNat x.numel b.dataPtr.toNat b.numel) &&
      decide (separate x.dataPtr.toNat x.numel i.dataPtr.toNat x.numel)) then .error .alias
  else .ok ()

def checkInputs (x b i : ArgumentMetadata) : Except InputError Unit :=
  match x, b, i with
  | .tensor x, .tensor b, .tensor i => checkTensorInputs x b i
  | _, _, _ => .error .tensors

def AdmittedArguments (x b i : ArgumentMetadata) : Prop :=
  match x, b, i with
  | .tensor x, .tensor b, .tensor i => InputContract x b i
  | _, _, _ => False

end VeriTile.Bench.Repairs.FusedActivation
