import VeriTile.Torch.Tensor
import bench.tritonbench_g.masked_add_cuda.Kernels

namespace VeriTile.Bench.Repairs.MaskedAddCuda

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

inductive AlphaArgument where
  | floating (bits : BitVec 64)
  | integer (value : Int)
  | other
  deriving DecidableEq, Repr

def supportedAlpha : AlphaArgument → Bool
  | .floating _ => Bool.true
  | .integer n => n == 0
  | .other => Bool.false

def maskDType (storage : MaskStorage) : VeriTile.Torch.DType :=
  match storage with
  | .int32 => .other "int32"
  | .int64 => .other "int64"

def maskBytes (storage : MaskStorage) : Nat :=
  match storage with
  | .int32 => 4
  | .int64 => 8

def selectedMask (m : TensorMetadata) : MaskStorage :=
  if m.dtype = .other "int64" then .int64 else .int32

def Separate (a bytesA b bytesB : Nat) : Prop :=
  a + bytesA ≤ b ∨ b + bytesB ≤ a

instance (a bytesA b bytesB : Nat) : Decidable (Separate a bytesA b bytesB) :=
  inferInstanceAs (Decidable (_ ∨ _))

/-- This is the checked intended API domain. Pointer conditions are queried
only for nonempty inputs; grad is disjoint from both read-only inputs. -/
def InputContract (g p m : TensorMetadata) (alpha : AlphaArgument) : Prop :=
  (g.layout = .strided ∧ p.layout = .strided ∧ m.layout = .strided) ∧
  (g.dtype = .floating .fp32 ∧ p.dtype = .floating .fp32 ∧
    (m.dtype = .other "int32" ∨ m.dtype = .other "int64")) ∧
  (g.device.kind = "cuda" ∧ p.device.kind = "cuda" ∧ m.device.kind = "cuda") ∧
  (g.device = p.device ∧ g.device = m.device) ∧
  (g.shape = p.shape ∧ g.shape = m.shape) ∧
  (g.isContiguous = Bool.true ∧ p.isContiguous = Bool.true ∧ m.isContiguous = Bool.true) ∧
  (g.isNeg = Bool.false ∧ p.isNeg = Bool.false ∧ m.isNeg = Bool.false) ∧
  supportedAlpha alpha = Bool.true ∧ g.numel < 2^31 ∧
  (0 < g.numel →
    (g.dataPtr.toNat % 4 = 0 ∧ p.dataPtr.toNat % 4 = 0 ∧
      m.dataPtr.toNat % maskBytes (selectedMask m) = 0) ∧
    Separate g.dataPtr.toNat (4*g.numel) p.dataPtr.toNat (4*g.numel) ∧
    Separate g.dataPtr.toNat (4*g.numel) m.dataPtr.toNat
      (maskBytes (selectedMask m)*g.numel))

inductive InputError where
  | tensors | layout | dtype | cuda | device | shape | contiguous | negative
  | alpha | extent | alignment | alias
  deriving DecidableEq, Repr

/-- Matches the Python guard order, including empty pointer-query bypass. -/
def checkTensorInputs (g p m : TensorMetadata) (alpha : AlphaArgument) : Except InputError Unit :=
  if g.layout != .strided || p.layout != .strided || m.layout != .strided then .error .layout
  else if g.dtype != .floating .fp32 || p.dtype != .floating .fp32 ||
      (m.dtype != .other "int32" && m.dtype != .other "int64") then .error .dtype
  else if !(g.isCuda && p.isCuda && m.isCuda) then .error .cuda
  else if g.device != p.device || g.device != m.device then .error .device
  else if g.shape != p.shape || g.shape != m.shape then .error .shape
  else if !(g.isContiguous && p.isContiguous && m.isContiguous) then .error .contiguous
  else if g.isNeg || p.isNeg || m.isNeg then .error .negative
  else if !supportedAlpha alpha then .error .alpha
  else if g.numel >= 2^31 then .error .extent
  else if g.numel = 0 then .ok ()
  else if g.dataPtr.toNat % 4 != 0 || p.dataPtr.toNat % 4 != 0 ||
      m.dataPtr.toNat % maskBytes (selectedMask m) != 0 then .error .alignment
  else if !(decide (Separate g.dataPtr.toNat (4*g.numel) p.dataPtr.toNat (4*g.numel)) &&
      decide (Separate g.dataPtr.toNat (4*g.numel) m.dataPtr.toNat
        (maskBytes (selectedMask m)*g.numel))) then .error .alias
  else .ok ()

def checkInputs (g p m : ArgumentMetadata) (alpha : AlphaArgument) : Except InputError Unit :=
  match g, p, m with
  | .tensor g, .tensor p, .tensor m => checkTensorInputs g p m alpha
  | _, _, _ => .error .tensors

def AdmittedArguments (g p m : ArgumentMetadata) (alpha : AlphaArgument) : Prop :=
  match g, p, m with
  | .tensor g, .tensor p, .tensor m => InputContract g p m alpha
  | _, _, _ => False

namespace Correct

/-- Independent in-place mathematical target. Every nonzero signed mask
vetoes the update; floating-point evaluation is specified separately. -/
noncomputable def formula (oldGrad data : ℝ) (mask : Int) (alpha : ℝ) : ℝ :=
  if mask = 0 then oldGrad + data * alpha else oldGrad

end Correct

end VeriTile.Bench.Repairs.MaskedAddCuda
