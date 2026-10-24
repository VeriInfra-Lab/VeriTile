import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.LayerNormTriton.Contract
open VeriTile.Torch

/-- Exact metadata guard; all FP32 words are admitted. Contiguous storage may
have a nonzero offset and arbitrary singleton/empty strides accepted by Torch. -/
def TensorContract (m : TensorMetadata) : Prop :=
  m.layout=.strided ∧ m.dtype=.floating .fp32 ∧ m.device.kind="cuda" ∧
  m.isContiguous=Bool.true ∧ m.isNeg=Bool.false ∧ (0<m.numel → m.dataPtr.toNat%4=0)

def tensorCheck (m : TensorMetadata) : Bool :=
  m.layout==.strided && m.dtype==.floating .fp32 && m.device.kind=="cuda" &&
  m.isContiguous && !m.isNeg && (!(m.numel>0) || m.dataPtr.toNat%4==0)

def EpsilonContract (value : Option Float) : Prop :=
  ∃ q,value=some q ∧ q.isNaN=Bool.false ∧ q.isInf=Bool.false ∧ 0.0<q

def epsilonCheck (value : Option Float) : Bool :=
  match value with
  | none => Bool.false
  | some q => !q.isNaN && !q.isInf && decide (0.0<q)

/-- None represents any invalid Python normalized_shape type or a negative,
non-integer dimension. A valid integer is represented by its singleton list. -/
def ShapeContract (x w b : TensorMetadata) (normalized : Option (List Nat)) : Prop :=
  ∃ ns,normalized=some ns ∧ ns≠[] ∧ ns.length≤x.shape.length ∧
    x.shape.drop (x.shape.length-ns.length)=ns ∧ w.shape=ns ∧ b.shape=ns

def shapeCheck (x w b : TensorMetadata) (normalized : Option (List Nat)) : Bool :=
  match normalized with
  | none => Bool.false
  | some ns => !ns.isEmpty && (ns.length≤x.shape.length) &&
    x.shape.drop (x.shape.length-ns.length)==ns && w.shape==ns && b.shape==ns

def InputContract (x w b : TensorMetadata) (normalized : Option (List Nat)) (eps : Option Float) : Prop :=
  TensorContract x ∧ TensorContract w ∧ TensorContract b ∧
  w.device=x.device ∧ b.device=x.device ∧ EpsilonContract eps ∧ ShapeContract x w b normalized

def inputCheck (x w b : TensorMetadata) (normalized : Option (List Nat)) (eps : Option Float) : Bool :=
  tensorCheck x && tensorCheck w && tensorCheck b && w.device==x.device && b.device==x.device &&
  epsilonCheck eps && shapeCheck x w b normalized

def GradientContract (dy x : TensorMetadata) (shape : List Nat) : Prop :=
  TensorContract dy ∧ dy.device=x.device ∧ dy.shape=shape

def gradientCheck (dy x : TensorMetadata) (shape : List Nat) : Bool :=
  tensorCheck dy && dy.device==x.device && dy.shape==shape

def checkArguments (x w b : ArgumentMetadata) (normalized : Option (List Nat)) (eps : Option Float) : Bool :=
  match x,w,b with
  | .tensor x,.tensor w,.tensor b => inputCheck x w b normalized eps
  | _,_,_ => Bool.false

def AdmittedArguments (x w b : ArgumentMetadata) (normalized : Option (List Nat)) (eps : Option Float) : Prop :=
  match x,w,b with
  | .tensor x,.tensor w,.tensor b => InputContract x w b normalized eps
  | _,_,_ => False

end VeriTile.Bench.Repairs.LayerNormTriton.Contract
