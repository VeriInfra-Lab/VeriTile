import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.LayerNormLiger.Contract
open VeriTile.Torch
noncomputable section

/-- Python's scalar classification and finiteness query; value is its Real
shadow. Its actual FP32 launch conversion remains a local numerical operation. -/
structure Epsilon where
  numeric : Bool
  finite : Bool
  value : ℝ

structure TensorContract (m : TensorMetadata) : Prop where
  layout : m.layout = .strided
  dtype : m.dtype = .floating .fp32
  cuda : m.device.kind = "cuda"
  noNeg : m.isNeg = Bool.false
  alignment : 0 < m.numel → m.dataPtr.toNat % 4 = 0

def width (x : TensorMetadata) : Nat := x.shape.getLastD 0
def rows (x : TensorMetadata) : Nat := x.shape.dropLast.prod

structure InputContract (x w b : TensorMetadata) (eps : Epsilon) : Prop where
  input : TensorContract x
  weight : TensorContract w
  bias : TensorContract b
  rank : 0 < x.shape.length
  nonempty : 0 < x.numel
  width_pos : 0 < width x
  width_bound : width x ≤ 65536
  weightShape : w.shape = [width x]
  biasShape : b.shape = [width x]
  device : w.device = x.device ∧ b.device = x.device
  epsilon : eps.numeric = Bool.true ∧ eps.finite = Bool.true ∧ 0 < eps.value

/-- The public autograd cotangent has the output's shape. The internal helper
also accepts another shape with the same element count and last dimension. -/
structure GradientContract (dy x : TensorMetadata) : Prop where
  tensor : TensorContract dy
  rank : 0 < dy.shape.length
  device : dy.device = x.device
  elements : dy.numel = x.numel
  columns : width dy = width x

inductive Error where
  | tensor | dtype | cuda | negative | alignment | shape | device | epsilon
  deriving DecidableEq, Repr

def checkTensor (m : TensorMetadata) : Except Error Unit :=
  if m.layout != .strided || m.dtype != .floating .fp32 then .error .dtype
  else if m.device.kind != "cuda" then .error .cuda
  else if m.isNeg then .error .negative
  else if m.numel > 0 && m.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()

def checkInputs (x w b : TensorMetadata) (eps : Epsilon) : Except Error Unit := by
  classical
  exact match checkTensor x,checkTensor w,checkTensor b with
  | .error e,_,_ => .error e
  | _,.error e,_ => .error e
  | _,_,.error e => .error e
  | .ok _,.ok _,.ok _ =>
    if x.shape.length == 0 || x.numel == 0 then .error .shape
    else if width x == 0 || width x > 65536 || w.shape != [width x] || b.shape != [width x] then .error .shape
    else if w.device != x.device || b.device != x.device then .error .device
    else if !eps.numeric || !eps.finite || decide (eps.value ≤ 0) then .error .epsilon
    else .ok ()

def checkGradient (dy x : TensorMetadata) : Except Error Unit :=
  match checkTensor dy with
  | .error e => .error e
  | .ok _ => if dy.shape.length == 0 || dy.device != x.device || dy.numel != x.numel || width dy != width x
    then .error .shape else .ok ()

def checkArguments (x w b : ArgumentMetadata) (eps : Epsilon) : Except Error Unit :=
  match x,w,b with
  | .tensor x,.tensor w,.tensor b => checkInputs x w b eps
  | _,_,_ => .error .tensor

def AdmittedArguments (x w b : ArgumentMetadata) (eps : Epsilon) : Prop :=
  match x,w,b with
  | .tensor x,.tensor w,.tensor b => InputContract x w b eps
  | _,_,_ => False

end
end VeriTile.Bench.Repairs.LayerNormLiger.Contract
