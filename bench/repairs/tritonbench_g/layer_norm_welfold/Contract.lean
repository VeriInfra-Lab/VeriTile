import VeriTile.Torch.FP32
import Mathlib.Analysis.SpecialFunctions.Sqrt
import VeriTile.Meta.StatementAudit

/-! Exact checked host domain of the selected FP32/CUDA0 source profile.
Storage binding and successful allocation are ordinary Tensor API laws, not
Python metadata checks and not numerical result assumptions. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Admission

structure Metadata where
  tensor : VeriTile.Torch.TensorMetadata
  strides : List Nat
  storageOffset : Nat
  deriving DecidableEq, Repr

inductive Argument where
  | other
  | tensor (value : Metadata)

inductive Error where
  | tensors | layout | dtype | device | rank | width | shape | strides | offset | negative | extent | alignment
  deriving DecidableEq, Repr

structure Contract (W Bias X : Metadata) (S D : Nat) : Prop where
  layout : W.tensor.layout = .strided ∧ Bias.tensor.layout = .strided ∧ X.tensor.layout = .strided
  dtype : W.tensor.dtype = .floating .fp32 ∧ Bias.tensor.dtype = .floating .fp32 ∧ X.tensor.dtype = .floating .fp32
  device : W.tensor.device = ⟨"cuda", 0⟩ ∧ Bias.tensor.device = ⟨"cuda", 0⟩ ∧ X.tensor.device = ⟨"cuda", 0⟩
  xShape : X.tensor.shape = [S,D]
  width : 0 < D
  affineShape : W.tensor.shape = [D] ∧ Bias.tensor.shape = [D]
  strides : X.strides = [D,1] ∧ W.strides = [1] ∧ Bias.strides = [1]
  offset : W.storageOffset = 0 ∧ Bias.storageOffset = 0 ∧ X.storageOffset = 0
  noNeg : W.tensor.isNeg = Bool.false ∧ Bias.tensor.isNeg = Bool.false ∧ X.tensor.isNeg = Bool.false
  extent : D + 2048 < 2^31 ∧ S * D + 2048 < 2^31
  alignment : (0 < S * D → X.tensor.dataPtr.toNat % 16 = 0) ∧
    W.tensor.dataPtr.toNat % 16 = 0 ∧ Bias.tensor.dataPtr.toNat % 16 = 0

def checkDimensions (W Bias X : Metadata) (S D : Nat) : Except Error (Nat × Nat) :=
  if D = 0 then .error .width
  else if W.tensor.shape != [D] || Bias.tensor.shape != [D] then .error .shape
  else if X.strides != [D,1] || W.strides != [1] || Bias.strides != [1] then .error .strides
  else if W.storageOffset != 0 || Bias.storageOffset != 0 || X.storageOffset != 0 then .error .offset
  else if W.tensor.isNeg || Bias.tensor.isNeg || X.tensor.isNeg then .error .negative
  else if D + 2048 >= 2^31 || S * D + 2048 >= 2^31 then .error .extent
  else if (S * D > 0 && X.tensor.dataPtr.toNat % 16 != 0) ||
      W.tensor.dataPtr.toNat % 16 != 0 || Bias.tensor.dataPtr.toNat % 16 != 0 then .error .alignment
  else .ok (S,D)

def checkTensors (W Bias X : Metadata) : Except Error (Nat × Nat) :=
  if W.tensor.layout != .strided || Bias.tensor.layout != .strided || X.tensor.layout != .strided then .error .layout
  else if W.tensor.dtype != .floating .fp32 || Bias.tensor.dtype != .floating .fp32 ||
      X.tensor.dtype != .floating .fp32 then .error .dtype
  else if W.tensor.device != ⟨"cuda", 0⟩ || Bias.tensor.device != ⟨"cuda", 0⟩ ||
      X.tensor.device != ⟨"cuda", 0⟩ then .error .device
  else match X.tensor.shape with
    | [S,D] => checkDimensions W Bias X S D
    | _ => .error .rank

def checkInputs (W Bias X : Argument) : Except Error (Nat × Nat) :=
  match W, Bias, X with
  | .tensor w, .tensor b, .tensor x => checkTensors w b x
  | _, _, _ => .error .tensors

def Admitted (W Bias X : Argument) (S D : Nat) : Prop :=
  match W, Bias, X with
  | .tensor w, .tensor b, .tensor x => Contract w b x S D
  | _, _, _ => False

end VeriTile.Bench.Repairs.LayerNormWelfold.Admission

/-! Independent mathematics over the original logical Tensor values. These
formulas have no kernel, reduction-tree, execution or native-graph dependency. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Math
noncomputable section

def mean (x : Nat → ℝ) (D row : Nat) : ℝ :=
  ((List.range D).map (fun col => x (col + D * row))).sum / D

def variance (x : Nat → ℝ) (D row : Nat) : ℝ :=
  ((List.range D).map (fun col => (x (col + D * row) - mean x D row)^2)).sum / D

def inverse (x : Nat → ℝ) (D row : Nat) : ℝ :=
  1 / Real.sqrt (variance x D row + 1e-05)

def output (x w bias : Nat → ℝ) (D row col : Nat) : ℝ :=
  ((x (col + D * row) - mean x D row) * inverse x D row) * w col + bias col

end
end VeriTile.Bench.Repairs.LayerNormWelfold.Math
