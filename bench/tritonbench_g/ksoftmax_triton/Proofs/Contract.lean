import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.Ksoftmax
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def batches (x : TensorMetadata) := x.shape[0]?.getD 0
def rows (x : TensorMetadata) := x.shape[1]?.getD 0
def columns (x : TensorMetadata) := x.shape[2]?.getD 0

def TensorContract (x : TensorMetadata) : Prop :=
  x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
  x.isContiguous = Bool.true ∧ x.isNeg = Bool.false

def ExtentContract (x : TensorMetadata) : Prop :=
  batches x < 2^31 ∧ rows x < 2^31 ∧ columns x ≤ 2^30 ∧ x.numel < 2^61

def separate (a na b nb : Nat) : Prop := a+4*na ≤ b ∨ b+4*nb ≤ a

def PairContract (dst input : TensorMetadata) : Prop :=
  input.dataPtr.toNat%4=0 ∧
  separate dst.dataPtr.toNat dst.numel input.dataPtr.toNat input.numel

def KnownMode (mode : Option String) : Prop := mode=none ∨ mode=some "qk" ∨ mode=some "bk"
def MaskContract (x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String) : Prop :=
  if mode=none then True else match mask with
  | .nonTensor => False
  | .tensor m => TensorContract m ∧ m.device=x.device ∧
      m.shape=[if mode=some "qk" then rows x else batches x,columns x]

def MaskMemory (dst : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String) : Prop :=
  if mode=none then True else match mask with
  | .nonTensor => False
  | .tensor m => PairContract dst m

def ForwardContract (y x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String) : Prop :=
  TensorContract y ∧ TensorContract x ∧ x.shape.length=3 ∧ y.shape=x.shape ∧
  y.device=x.device ∧ KnownMode mode ∧ MaskContract x mask mode ∧
  (0<x.numel → ExtentContract x ∧ y.dataPtr.toNat%4=0 ∧ PairContract y x ∧ MaskMemory y mask mode)

def BackwardContract (dst grad saved : TensorMetadata) : Prop :=
  TensorContract dst ∧ TensorContract grad ∧ TensorContract saved ∧
  saved.shape.length=3 ∧ dst.shape=saved.shape ∧ grad.shape=saved.shape ∧
  dst.device=saved.device ∧ grad.device=saved.device ∧
  (0<saved.numel → ExtentContract saved ∧ dst.dataPtr.toNat%4=0 ∧
    PairContract dst grad ∧ PairContract dst saved)

instance (x : TensorMetadata) : Decidable (TensorContract x) := inferInstanceAs (Decidable (_ ∧ _))
instance (x : TensorMetadata) : Decidable (ExtentContract x) := inferInstanceAs (Decidable (_ ∧ _))
instance (a na b nb : Nat) : Decidable (separate a na b nb) := inferInstanceAs (Decidable (_ ∨ _))
instance (x y : TensorMetadata) : Decidable (PairContract x y) := inferInstanceAs (Decidable (_ ∧ _))
instance (mode : Option String) : Decidable (KnownMode mode) := inferInstanceAs (Decidable (_ ∨ _))
instance (x : TensorMetadata) (m : ArgumentMetadata) (mode : Option String) : Decidable (MaskContract x m mode) := by
  unfold MaskContract; split <;> cases m <;> infer_instance
instance (x : TensorMetadata) (m : ArgumentMetadata) (mode : Option String) : Decidable (MaskMemory x m mode) := by
  unfold MaskMemory; split <;> cases m <;> infer_instance

inductive InputError where | tensor | flags | metadata | shape | device | mode | mask | extent | memory
  deriving DecidableEq, Repr

/-- `flags` records exact Python bool classification for LOG and CAUSAL. -/
def checkForward (y x mask : ArgumentMetadata) (mode : Option String) (flags : Bool) : Except InputError Unit :=
  if !flags then .error .flags else
  if ¬KnownMode mode then .error .mode else
  match y,x with
  | .tensor y,.tensor x =>
    if ¬(TensorContract y ∧ TensorContract x) then .error .metadata
    else if ¬(x.shape.length=3 ∧ y.shape=x.shape) then .error .shape
    else if y.device≠x.device then .error .device
    else if ¬MaskContract x mask mode then .error .mask
    else if x.numel=0 then .ok ()
    else if ¬ExtentContract x then .error .extent
    else if ¬(y.dataPtr.toNat%4=0 ∧ PairContract y x ∧ MaskMemory y mask mode) then .error .memory
    else .ok ()
  | _,_ => .error .tensor

def checkBackward (dst grad saved : ArgumentMetadata) (flags : Bool) : Except InputError Unit :=
  if !flags then .error .flags else match dst,grad,saved with
  | .tensor dst,.tensor grad,.tensor saved =>
    if ¬(TensorContract dst ∧ TensorContract grad ∧ TensorContract saved) then .error .metadata
    else if ¬(saved.shape.length=3 ∧ dst.shape=saved.shape ∧ grad.shape=saved.shape) then .error .shape
    else if ¬(dst.device=saved.device ∧ grad.device=saved.device) then .error .device
    else if saved.numel=0 then .ok ()
    else if ¬ExtentContract saved then .error .extent
    else if ¬(dst.dataPtr.toNat%4=0 ∧ PairContract dst grad ∧ PairContract dst saved) then .error .memory
    else .ok ()
  | _,_,_ => .error .tensor

end VeriTile.Bench.Repairs.Ksoftmax
