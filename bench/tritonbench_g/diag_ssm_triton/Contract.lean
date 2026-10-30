import VeriTile.Torch.Tensor

/-! Exact supported metadata domain. Complex64 retains its explicit source
precision and is represented by the canonical Torch dtype tag; its two kernel
components are FP32. Tensor ownership and the API's storage binding are separate
ordinary runtime facts, not extra admission conditions. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Contract

structure Metadata extends VeriTile.Torch.TensorMetadata where
  isConj : Bool
  deriving DecidableEq, Repr

inductive Argument where
  | nonTensor
  | tensor (metadata : Metadata)
  deriving DecidableEq, Repr

def SupportedDType (m : Metadata) : Prop :=
  m.dtype = .floating .fp32 ∨ m.dtype = .other "complex64"

structure TensorContract (contiguous : Bool) (m : Metadata) : Prop where
  layout : m.layout = .strided
  dtype : SupportedDType m
  cuda : m.device.kind = "cuda"
  dense : contiguous = Bool.true → m.isContiguous = Bool.true
  resolved : m.isNeg = Bool.false ∧ m.isConj = Bool.false
  alignment : 0 < m.numel → m.dataPtr.toNat % 4 = 0

structure InputContract (s x transition : Metadata) : Prop where
  initial : TensorContract Bool.true s
  input : TensorContract Bool.true x
  parameter : TensorContract Bool.true transition
  dtype : s.dtype = x.dtype ∧ x.dtype = transition.dtype
  device : s.device = x.device ∧ x.device = transition.device
  rank : x.shape.length = 3
  initialShape : s.shape = x.shape.drop 1
  parameterShape : transition.shape = x.shape.drop 2

/-- The original backward already calls contiguous(). This domain preserves
that behavior and admits arbitrary incoming cotangent strides, but no lazy
negative/conjugate flags or additional materialization of forward inputs. -/
structure GradientContract (gy y : Metadata) : Prop where
  tensor : TensorContract Bool.false gy
  dtype : gy.dtype = y.dtype
  device : gy.device = y.device
  shape : gy.shape = y.shape

inductive Error where
  | tensor | layout | dtype | cuda | contiguous | resolved | alignment | device | shape
  deriving DecidableEq, Repr

def checkTensor (contiguous : Bool) (m : Metadata) : Except Error Unit :=
  if m.layout != .strided then .error .layout
  else if !(m.dtype == .floating .fp32 || m.dtype == .other "complex64") then .error .dtype
  else if m.device.kind != "cuda" then .error .cuda
  else if contiguous && !m.isContiguous then .error .contiguous
  else if m.isNeg || m.isConj then .error .resolved
  else if m.numel > 0 && m.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()

def checkInputs (s x transition : Metadata) : Except Error Unit :=
  match checkTensor Bool.true s, checkTensor Bool.true x, checkTensor Bool.true transition with
  | .error e, _, _ => .error e
  | _, .error e, _ => .error e
  | _, _, .error e => .error e
  | .ok _, .ok _, .ok _ =>
    if s.dtype != x.dtype || x.dtype != transition.dtype then .error .dtype
    else if s.device != x.device || x.device != transition.device then .error .device
    else if x.shape.length != 3 then .error .shape
    else if s.shape != x.shape.drop 1 || transition.shape != x.shape.drop 2 then .error .shape
    else .ok ()

def checkGradient (gy y : Metadata) : Except Error Unit :=
  match checkTensor Bool.false gy with
  | .error e => .error e
  | .ok _ =>
    if gy.dtype != y.dtype || gy.device != y.device || gy.shape != y.shape then .error .shape
    else .ok ()

def checkArguments (s x transition : Argument) : Except Error Unit :=
  match s, x, transition with
  | .tensor s, .tensor x, .tensor transition => checkInputs s x transition
  | _, _, _ => .error .tensor

def AdmittedArguments (s x transition : Argument) : Prop :=
  match s, x, transition with
  | .tensor s, .tensor x, .tensor transition => InputContract s x transition
  | _, _, _ => False

end VeriTile.Bench.Repairs.DiagSsmTriton.Contract
