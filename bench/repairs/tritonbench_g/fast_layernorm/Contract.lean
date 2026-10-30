import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.FastLayernorm.Contract
open VeriTile.Torch
noncomputable section

structure Epsilon where
  numeric : Bool
  finite : Bool
  value : ℝ

structure TensorContract (m : TensorMetadata) (contiguous : Bool) : Prop where
  layout : m.layout = .strided
  dtype : m.dtype = .floating .fp32
  cuda : m.device.kind = "cuda"
  dense : contiguous=Bool.true → m.isContiguous=Bool.true
  noNeg : m.isNeg = Bool.false
  alignment : 0<m.numel → m.dataPtr.toNat%4=0

def width (x : TensorMetadata) := x.shape.getLastD 0

structure InputContract (x w : TensorMetadata) (bias : Option TensorMetadata) (eps : Epsilon) : Prop where
  input : TensorContract x Bool.true
  weight : TensorContract w Bool.true
  biasTensor : ∀ b,bias=some b → TensorContract b Bool.true
  rank : 0<x.shape.length
  positive : 0<width x
  bounded : width x≤65536
  weightDevice : w.device=x.device
  biasDevice : ∀ b,bias=some b → b.device=x.device
  weightCount : width x≤w.numel
  biasCount : ∀ b,bias=some b → width x≤b.numel
  epsilon : eps.numeric=Bool.true ∧ eps.finite=Bool.true ∧ 0<eps.value

structure GradientContract (dy x : TensorMetadata) : Prop where
  tensor : TensorContract dy Bool.false
  device : dy.device=x.device
  shape : dy.shape=x.shape

inductive Error where
  | tensor | dtype | cuda | contiguous | negative | alignment | shape | device | epsilon
  deriving DecidableEq, Repr

def checkTensor (m : TensorMetadata) (contiguous : Bool) : Except Error Unit :=
  if m.layout != .strided || m.dtype != .floating .fp32 then .error .dtype
  else if m.device.kind != "cuda" then .error .cuda
  else if contiguous && !m.isContiguous then .error .contiguous
  else if m.isNeg then .error .negative
  else if m.numel>0 && m.dataPtr.toNat%4 != 0 then .error .alignment
  else .ok ()

def checkOptional (b : Option TensorMetadata) : Except Error Unit :=
  match b with
  | none => .ok ()
  | some b => checkTensor b Bool.true

def checkInputs (x w : TensorMetadata) (bias : Option TensorMetadata) (eps : Epsilon) : Except Error Unit := by
  classical
  exact match checkTensor x Bool.true,checkTensor w Bool.true,checkOptional bias with
  | .error e,_,_ => .error e
  | _,.error e,_ => .error e
  | _,_,.error e => .error e
  | .ok _,.ok _,.ok _ =>
    if x.shape.length==0 || width x==0 || width x>65536 then .error .shape
    else if w.device != x.device || (bias.map (fun b => b.device != x.device)).getD Bool.false then .error .device
    else if w.numel<width x || (bias.map (fun b => decide (b.numel<width x))).getD Bool.false then .error .shape
    else if !eps.numeric || !eps.finite || decide (eps.value≤0) then .error .epsilon
    else .ok ()

def checkGradient (dy x : TensorMetadata) : Except Error Unit :=
  match checkTensor dy Bool.false with
  | .error e => .error e
  | .ok _ => if dy.device != x.device || dy.shape != x.shape then .error .shape else .ok ()

def checkArguments (x w : ArgumentMetadata) (bias : Option ArgumentMetadata) (eps : Epsilon) : Except Error Unit :=
  match x,w,bias with
  | .tensor x,.tensor w,none => checkInputs x w none eps
  | .tensor x,.tensor w,some (.tensor b) => checkInputs x w (some b) eps
  | _,_,_ => .error .tensor

def AdmittedArguments (x w : ArgumentMetadata) (bias : Option ArgumentMetadata) (eps : Epsilon) : Prop :=
  match x,w,bias with
  | .tensor x,.tensor w,none => InputContract x w none eps
  | .tensor x,.tensor w,some (.tensor b) => InputContract x w (some b) eps
  | _,_,_ => False

end
end VeriTile.Bench.Repairs.FastLayernorm.Contract
