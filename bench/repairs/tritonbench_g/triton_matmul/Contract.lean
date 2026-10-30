import VeriTile.Torch.Tensor
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Contract
open VeriTile.Torch

def rows (t : TensorMetadata) := t.shape[0]?.getD 0
def columns (t : TensorMetadata) := t.shape[1]?.getD 0
def bytes (t : TensorMetadata) := if t.dtype=.other "float16" then 2 else 1
def Supported (t : TensorMetadata) : Prop :=
  t.dtype=.other "float16" ∨ t.dtype=.other "float8_e4m3fn"
def Empty (a b : TensorMetadata) : Prop := rows a=0 ∨ columns b=0
def Dimensions (a b : TensorMetadata) : Prop :=
  rows a<2^63 ∧ columns b<2^63 ∧ columns a<2^63
/-- Bounds for the actual signed64 element-to-byte pointer increments and
the signed32 program-id widened by the source. These are checked explicitly. -/
def Envelope (a b : TensorMetadata) : Prop :=
  bytes a*(rows a*columns a)<2^63 ∧ bytes a*(columns a*columns b)<2^63 ∧
  bytes a*(rows a*columns b)<2^63 ∧ ((rows a+127)/128)*((columns b+255)/256)<2^31
def Window (t : TensorMetadata) : Prop :=
  t.numel=0 ∨ (t.dataPtr.toNat%bytes t=0 ∧ t.dataPtr.toNat+bytes t*t.numel≤2^64)
def Execution (a b : TensorMetadata) : Prop := Empty a b ∨ (Envelope a b ∧ Window a ∧ Window b)

structure InputContract (a b : TensorMetadata) : Prop where
  layout : a.layout=.strided ∧ b.layout=.strided
  dtype : Supported a ∧ b.dtype=a.dtype
  device : a.device.kind="cuda" ∧ b.device.kind="cuda" ∧ b.device=a.device
  rank : a.shape.length=2 ∧ b.shape.length=2
  inner : columns a=rows b
  contiguous : a.isContiguous=true ∧ b.isContiguous=true
  resolved : a.isNeg=false ∧ b.isNeg=false
  dimensions : Dimensions a b
  execution : Execution a b

inductive Error where
  | tensors | layout | dtype | device | rank | inner | contiguous | negative | dimensions | envelope | window
  deriving DecidableEq,Repr
instance (t : TensorMetadata) : Decidable (Supported t) := inferInstanceAs (Decidable (_ ∨ _))
instance (a b : TensorMetadata) : Decidable (Empty a b) := inferInstanceAs (Decidable (_ ∨ _))
instance (a b : TensorMetadata) : Decidable (Dimensions a b) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))
instance (a b : TensorMetadata) : Decidable (Envelope a b) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))
instance (t : TensorMetadata) : Decidable (Window t) := inferInstanceAs (Decidable (_ ∨ (_ ∧ _)))

def checkTensorInputs (a b : TensorMetadata) : Except Error Unit :=
  if ¬(a.layout=.strided ∧ b.layout=.strided) then .error .layout
  else if ¬(Supported a ∧ b.dtype=a.dtype) then .error .dtype
  else if ¬(a.device.kind="cuda" ∧ b.device.kind="cuda" ∧ b.device=a.device) then .error .device
  else if ¬(a.shape.length=2 ∧ b.shape.length=2) then .error .rank
  else if ¬columns a=rows b then .error .inner
  else if ¬(a.isContiguous=true ∧ b.isContiguous=true) then .error .contiguous
  else if ¬(a.isNeg=false ∧ b.isNeg=false) then .error .negative
  else if ¬Dimensions a b then .error .dimensions
  else if Empty a b then .ok ()
  else if ¬Envelope a b then .error .envelope
  else if ¬(Window a ∧ Window b) then .error .window
  else .ok ()
def checkInputs (a b : ArgumentMetadata) : Except Error Unit :=
  match a,b with
  | .tensor a,.tensor b => checkTensorInputs a b
  | _,_ => .error .tensors
def AdmittedArguments (a b : ArgumentMetadata) : Prop :=
  match a,b with
  | .tensor a,.tensor b => InputContract a b
  | _,_ => False
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Contract
