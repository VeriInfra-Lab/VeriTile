import VeriTile.Torch.Tensor
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SourceIndex

namespace VeriTile.Bench.Repairs.Conv2d
open VeriTile.Torch

/-- Actual entry-query metadata. Storage bases and capacities are bytes;
`dataPtr` is the queried tensor pointer, including any actual storage offset. -/
structure TensorView where
  identity : Nat
  tensor : TensorMetadata
  strides : List Int
  isConj : Bool
  storageBase : Nat
  capacity : Nat

def TensorView.pointer (t : TensorView) := t.tensor.dataPtr.toNat

def TensorKind (t : TensorView) : Prop :=
  t.tensor.layout=.strided ∧ t.tensor.dtype=.floating .fp32 ∧
  t.tensor.device.kind="cuda" ∧ t.tensor.isNeg=Bool.false ∧ t.isConj=Bool.false
instance (t : TensorView) : Decidable (TensorKind t) := by unfold TensorKind;infer_instance

def Stored (t : TensorView) (words : Nat) : Prop :=
  0<words → t.pointer%4=0 ∧ t.storageBase≤t.pointer ∧
    t.pointer+4*words≤t.storageBase+t.capacity ∧ t.storageBase+t.capacity≤2^64
instance (t : TensorView) (words : Nat) : Decidable (Stored t words) := by unfold Stored;infer_instance

def inputWords (s : Shape) := if s.n=0 then 0 else inputTop s+1
def weightWords (s : Shape) := weightTop s+1

/-- The `Shape` record is the normalized successful query record: dimensions,
strides, integer arguments and computed quotients/output extents. Its binding
is checked below, including the two exact Python floor-division formulas. -/
structure Call where
  shape : Shape
  input : TensorView
  weight : TensorView
  flags : Flags
  callerDevice : Nat

/-- Every field of `Valid`, spelled as the executable guard's finite checks. -/
def validCheck (s : Shape) : Bool :=
  decide (0<s.cg ∧ 0<s.og ∧ 0<s.h ∧ 0<s.w ∧ 0<s.kh ∧ 0<s.kw ∧
    0<s.sh ∧ 0<s.sw ∧ 0<s.groups ∧ 0<s.oh ∧ 0<s.ow ∧
    (s.oh:Int)=((s.h:Int)+2*s.ph-s.kh)/s.sh+1 ∧
    (s.ow:Int)=((s.w:Int)+2*s.pw-s.kw)/s.sw+1 ∧
    0<s.sn ∧ 0<s.sc ∧ 0<s.sx ∧ 0<s.sy ∧ 0<s.so ∧ 0<s.tc ∧ 0<s.tx ∧ 0<s.ty)

def MetadataBinding (c : Call) : Prop :=
  c.input.tensor.shape=[c.shape.n,c.shape.channels,c.shape.h,c.shape.w] ∧
  c.weight.tensor.shape=[c.shape.outputs,c.shape.cg,c.shape.kh,c.shape.kw] ∧
  c.input.strides=[c.shape.sn,c.shape.sc,c.shape.sx,c.shape.sy] ∧
  c.weight.strides=[c.shape.so,c.shape.tc,c.shape.tx,c.shape.ty]
instance (c : Call) : Decidable (MetadataBinding c) := by unfold MetadataBinding;infer_instance

def LaunchGrid (s : Shape) : Prop :=
  0<s.n → blocks s.outputs 32≤65535 ∧ s.groups≤65535
instance (s : Shape) : Decidable (LaunchGrid s) := by unfold LaunchGrid;infer_instance

structure Domain (c : Call) : Prop where
  valid : Valid c.shape
  inputKind : TensorKind c.input
  weightKind : TensorKind c.weight
  binding : MetadataBinding c
  sameDevice : c.input.tensor.device=c.weight.tensor.device
  currentDevice : c.callerDevice=c.input.tensor.device.index
  inputStorage : Stored c.input (inputWords c.shape)
  weightStorage : Stored c.weight (weightWords c.shape)
  signed : envelope c.shape<2^31
  grid : LaunchGrid c.shape

inductive GuardError where
  | object | scalarType | scalarRange | metadata | shape | device | storage | envelope | grid
  deriving DecidableEq

def checkCall (c : Call) : Except GuardError Unit :=
  if ¬TensorKind c.input ∨ ¬TensorKind c.weight then .error .metadata
  else if validCheck c.shape≠Bool.true ∨ ¬MetadataBinding c then .error .shape
  else if c.input.tensor.device≠c.weight.tensor.device ∨
      c.callerDevice≠c.input.tensor.device.index then .error .device
  else if ¬Stored c.input (inputWords c.shape) ∨ ¬Stored c.weight (weightWords c.shape) then .error .storage
  else if envelope c.shape≥2^31 then .error .envelope
  else if ¬LaunchGrid c.shape then .error .grid else .ok ()

/-- Python type classification precedes the normalized natural query record.
Integers are strict `type(v) is int`, flags strict `type(v) is bool`.
Negative kernel/stride/group or padding arguments cannot enter `Call`. -/
inductive ClassifiedCall where
  | otherObject | otherScalarType | negativeScalar | normalized (call : Call)

def check : ClassifiedCall → Except GuardError Unit
  | .otherObject => .error .object
  | .otherScalarType => .error .scalarType
  | .negativeScalar => .error .scalarRange
  | .normalized c => checkCall c

def Admitted : ClassifiedCall → Prop
  | .normalized c => Domain c
  | _ => False

end VeriTile.Bench.Repairs.Conv2d
