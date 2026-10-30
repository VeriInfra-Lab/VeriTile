import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Contract
open VeriTile.Torch

def rows (t : TensorMetadata) := t.shape[0]?.getD 0
def columns (t : TensorMetadata) := t.shape[1]?.getD 0

structure Config where
  (blockM blockN blockK groupM warps stages ctas : Nat)
  deriving DecidableEq, Repr

/-- Classification of the Python Config object, exact four-key dictionary,
and builtin integer fields. Negative fields cannot satisfy the later positive
requirements and are classified as invalid. This is not a kernel premise. -/
inductive ConfigArgument where
  | invalid
  | config (value : Config)
  deriving DecidableEq, Repr

def ConfigContract (p : Config) : Prop :=
  (16 ≤ p.blockM ∧ 16 ≤ p.blockN ∧ 32 ≤ p.blockK) ∧
  (p.blockM &&& (p.blockM-1) = 0 ∧ p.blockN &&& (p.blockN-1) = 0 ∧
    p.blockK &&& (p.blockK-1) = 0) ∧
  (p.blockM*p.blockN ≤ 2^20 ∧ p.blockM*p.blockK ≤ 2^20 ∧ p.blockK*p.blockN ≤ 2^20) ∧
  0 < p.groupM ∧ p.warps ∈ [1,2,4,8,16,32] ∧ 0 < p.stages ∧ p.ctas = 1

def ceil (n b : Nat) := (n+b-1)/b
def padded (n b : Nat) := ceil n b*b

/-- Every expression here is evaluated by the actual signed-int32 source:
cdiv numerators, grouped width/grid, padded pointer coordinates and the final
K pointer advance. These are checked by Python, never hidden in storage laws. -/
def indexExpressions (M N K : Nat) (p : Config) : List Nat :=
  [M+p.blockM-1, N+p.blockN-1, K+p.blockK-1,
   p.groupM*ceil N p.blockN, ceil M p.blockM*ceil N p.blockN,
   padded M p.blockM*K+padded K p.blockK+p.blockK,
   (padded K p.blockK+p.blockK)*N+padded N p.blockN,
   padded M p.blockM*N+padded N p.blockN]

def IndexEnvelope (M N K : Nat) (p : Config) : Prop :=
  ∀ v ∈ indexExpressions M N K p, v < 2^31

def ByteWindow (t : TensorMetadata) (bytes : Nat) : Prop :=
  t.numel = 0 ∨ (t.dataPtr.toNat % bytes = 0 ∧ t.dataPtr.toNat+bytes*t.numel ≤ 2^64)

def Separate (input : TensorMetadata) (bytes : Nat) (output : TensorMetadata) : Prop :=
  input.numel = 0 ∨ output.numel = 0 ∨
  input.dataPtr.toNat+bytes*input.numel ≤ output.dataPtr.toNat ∨
  output.dataPtr.toNat+4*output.numel ≤ input.dataPtr.toNat

def tensors (scaled : Bool) (a b c scale : TensorMetadata) : List TensorMetadata :=
  if scaled then [a,b,c,scale] else [a,b,c]

def Layouts (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ∀ t ∈ tensors scaled a b c s, t.layout = .strided
def DTypes (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  a.dtype = .other "int8" ∧ b.dtype = .other "int8" ∧ c.dtype = .other "int32" ∧
  (scaled = true → s.dtype = .floating .fp32)
def CUDA (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ∀ t ∈ tensors scaled a b c s, t.device.kind = "cuda"
def SameDevice (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ∀ t ∈ tensors scaled a b c s, t.device = a.device
def Ranks (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ∀ t ∈ tensors scaled a b c s, t.shape.length = 2
def Shapes (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  columns a = rows b ∧ c.shape = [rows a,columns b] ∧ (scaled = true → s.shape = [rows a,1])
def Contiguous (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ∀ t ∈ tensors scaled a b c s, t.isContiguous = true
def Resolved (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ∀ t ∈ tensors scaled a b c s, t.isNeg = false
def Empty (a b : TensorMetadata) : Prop := rows a = 0 ∨ columns b = 0
def Windows (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  ByteWindow a 1 ∧ ByteWindow b 1 ∧ ByteWindow c 4 ∧ (scaled = true → ByteWindow s 4)
def Nonoverlap (scaled : Bool) (a b c s : TensorMetadata) : Prop :=
  Separate a 1 c ∧ Separate b 1 c ∧ (scaled = true → Separate s 4 c)
def ExecutionContract (scaled : Bool) (a b c s : TensorMetadata) (p : Config) : Prop :=
  Empty a b ∨ (IndexEnvelope (rows a) (columns b) (columns a) p ∧
    Windows scaled a b c s ∧ Nonoverlap scaled a b c s)

structure InputContract (scaled : Bool) (a b c s : TensorMetadata) (p : Config) : Prop where
  layouts : Layouts scaled a b c s
  dtypes : DTypes scaled a b c s
  cuda : CUDA scaled a b c s
  device : SameDevice scaled a b c s
  ranks : Ranks scaled a b c s
  shapes : Shapes scaled a b c s
  contiguous : Contiguous scaled a b c s
  resolved : Resolved scaled a b c s
  config : ConfigContract p
  execution : ExecutionContract scaled a b c s p

inductive Error where
  | tensors | layout | dtype | cuda | device | rank | shapes | contiguous | negative
  | config | envelope | window | overlap
  deriving DecidableEq, Repr

instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Layouts s a b c d) :=
  inferInstanceAs (Decidable (∀ t ∈ tensors s a b c d, t.layout = .strided))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (DTypes s a b c d) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (CUDA s a b c d) :=
  inferInstanceAs (Decidable (∀ t ∈ tensors s a b c d, t.device.kind = "cuda"))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (SameDevice s a b c d) :=
  inferInstanceAs (Decidable (∀ t ∈ tensors s a b c d, t.device = a.device))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Ranks s a b c d) :=
  inferInstanceAs (Decidable (∀ t ∈ tensors s a b c d, t.shape.length = 2))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Shapes s a b c d) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Contiguous s a b c d) :=
  inferInstanceAs (Decidable (∀ t ∈ tensors s a b c d, t.isContiguous = true))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Resolved s a b c d) :=
  inferInstanceAs (Decidable (∀ t ∈ tensors s a b c d, t.isNeg = false))
instance (p : Config) : Decidable (ConfigContract p) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
instance (a b : TensorMetadata) : Decidable (Empty a b) := inferInstanceAs (Decidable (_ ∨ _))
instance (M N K : Nat) (p : Config) : Decidable (IndexEnvelope M N K p) :=
  inferInstanceAs (Decidable (∀ v ∈ indexExpressions M N K p, v < 2^31))
instance (t : TensorMetadata) (bytes : Nat) : Decidable (ByteWindow t bytes) :=
  inferInstanceAs (Decidable (_ ∨ (_ ∧ _)))
instance (t : TensorMetadata) (bytes : Nat) (c : TensorMetadata) : Decidable (Separate t bytes c) :=
  inferInstanceAs (Decidable (_ ∨ _ ∨ _ ∨ _))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Windows s a b c d) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))
instance (s : Bool) (a b c d : TensorMetadata) : Decidable (Nonoverlap s a b c d) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def checkTensorInputs (scaled : Bool) (a b c s : TensorMetadata) (p : Config) : Except Error Unit :=
  if ¬Layouts scaled a b c s then .error .layout
  else if ¬DTypes scaled a b c s then .error .dtype
  else if ¬CUDA scaled a b c s then .error .cuda
  else if ¬SameDevice scaled a b c s then .error .device
  else if ¬Ranks scaled a b c s then .error .rank
  else if ¬Shapes scaled a b c s then .error .shapes
  else if ¬Contiguous scaled a b c s then .error .contiguous
  else if ¬Resolved scaled a b c s then .error .negative
  else if ¬ConfigContract p then .error .config
  else if Empty a b then .ok ()
  else if ¬IndexEnvelope (rows a) (columns b) (columns a) p then .error .envelope
  else if ¬Windows scaled a b c s then .error .window
  else if ¬Nonoverlap scaled a b c s then .error .overlap
  else .ok ()

def checkInputs (scaled : Bool) (a b c s : ArgumentMetadata) (p : ConfigArgument) : Except Error Unit :=
  match a,b,c,p with
  | .tensor a,.tensor b,.tensor c,.config p =>
    if scaled then match s with
      | .tensor s => checkTensorInputs true a b c s p
      | _ => .error .tensors
    else checkTensorInputs false a b c a p
  | .tensor _,.tensor _,.tensor _,.invalid => .error .config
  | _,_,_,_ => .error .tensors

def AdmittedArguments (scaled : Bool) (a b c s : ArgumentMetadata) (p : ConfigArgument) : Prop :=
  match a,b,c,p with
  | .tensor a,.tensor b,.tensor c,.config p =>
    if scaled then match s with
      | .tensor s => InputContract true a b c s p
      | _ => False
    else InputContract false a b c a p
  | _,_,_,_ => False

end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Contract
