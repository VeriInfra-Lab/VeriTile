import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Contract
open VeriTile.Torch
noncomputable section

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

structure InputContract (x w : TensorMetadata) (eps : Epsilon) : Prop where
  input : TensorContract x
  weight : TensorContract w
  rank : x.shape.length = 3
  weightRank : w.shape.length = 2
  weightShape : w.shape = x.shape.drop 1
  device : w.device = x.device
  epsilon : eps.numeric = Bool.true ∧ eps.finite = Bool.true ∧ 0 < eps.value

inductive Error where
  | tensor | dtype | cuda | negative | alignment | shape | device | epsilon
  deriving DecidableEq, Repr

def checkTensor (m : TensorMetadata) : Except Error Unit :=
  if m.layout != .strided || m.dtype != .floating .fp32 then .error .dtype
  else if m.device.kind != "cuda" then .error .cuda
  else if m.isNeg then .error .negative
  else if m.numel > 0 && m.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()

def checkInputs (x w : TensorMetadata) (eps : Epsilon) : Except Error Unit := by
  classical
  exact match checkTensor x,checkTensor w with
  | .error e,_ => .error e
  | _,.error e => .error e
  | .ok _,.ok _ =>
    if x.shape.length != 3 || w.shape.length != 2 || w.shape != x.shape.drop 1 then .error .shape
    else if w.device != x.device then .error .device
    else if !eps.numeric || !eps.finite || decide (eps.value ≤ 0) then .error .epsilon
    else .ok ()

def checkArguments (x w : ArgumentMetadata) (eps : Epsilon) : Except Error Unit :=
  match x,w with
  | .tensor x,.tensor w => checkInputs x w eps
  | _,_ => .error .tensor

def AdmittedArguments (x w : ArgumentMetadata) (eps : Epsilon) : Prop :=
  match x,w with
  | .tensor x,.tensor w => InputContract x w eps
  | _,_ => False

/- Independent full logical feature-axis formulas. These do not invoke the
source, a reduction schedule, a memory interpreter or FP decoding. -/
namespace Math
def mean (N : Nat) (x : Nat → ℝ) : ℝ := (∑ j : Fin N, x j.val)/(N:ℝ)
def variance (N : Nat) (x : Nat → ℝ) : ℝ :=
  (∑ j : Fin N, (x j.val-mean N x)^2)/(N:ℝ)
def inverse (N : Nat) (x : Nat → ℝ) (eps : ℝ) : ℝ :=
  (Real.sqrt (variance N x+eps))⁻¹
def output (N : Nat) (x w : Nat → ℝ) (eps : ℝ) (col : Nat) : ℝ :=
  ((x col-mean N x)*inverse N x eps)*w col
end Math

end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Contract
