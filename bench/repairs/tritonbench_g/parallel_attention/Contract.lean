import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.parallel_attention.Proofs.MachineBounds
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostAPI

namespace VeriTile.Bench.Repairs.ParallelAttention.Checked
open VeriTile.Torch

/-- Actual Torch query record. Both addresses and storage capacity are bytes;
`dataPtr` already includes the tensor's storage offset. -/
structure TensorView where
  identity : Nat
  tensor : VeriTile.Torch.TensorMetadata
  strides : List Int
  isConj : Bool
  storageBase : Nat
  capacity : Nat

def TensorView.pointer (t : TensorView) := t.tensor.dataPtr.toNat

def TensorKind (t : TensorView) : Prop :=
  t.tensor.layout=.strided ∧ t.tensor.dtype=.floating .fp32 ∧
  t.tensor.device.kind="cuda" ∧ t.tensor.isContiguous=true ∧
  t.tensor.isNeg=false ∧ t.isConj=false
instance (t : TensorView) : Decidable (TensorKind t) := by unfold TensorKind;infer_instance

def Stored (t : TensorView) (words : Nat) : Prop :=
  0<words → t.pointer%4=0 ∧ t.storageBase≤t.pointer ∧
    t.pointer+4*words≤t.storageBase+t.capacity ∧ t.storageBase+t.capacity≤2^63
instance (t : TensorView) (words : Nat) : Decidable (Stored t words) := by unfold Stored;infer_instance

structure Call where
  shape : Shape
  q : TensorView
  k : TensorView
  v : TensorView
  callerDevice : Nat
  normalize : Bool
  returnBoth : Bool

def MetadataBinding (c : Call) : Prop :=
  c.q.tensor.shape=[c.shape.B,c.shape.H,c.shape.T,c.shape.K] ∧
  c.k.tensor.shape=[c.shape.B,c.shape.H,c.shape.T,c.shape.K] ∧
  c.v.tensor.shape=[c.shape.B,c.shape.H,c.shape.T,c.shape.V]
instance (c : Call) : Decidable (MetadataBinding c) := by unfold MetadataBinding;infer_instance

def shapeCheck (s : Shape) : Bool := decide
  (0<s.K ∧ s.K≤128 ∧ 0<s.V ∧ s.BK=selectedTile s.K ∧ s.BV=selectedTile s.V)

def machineCheck (s : Shape) : Bool := decide
  (s.B=0 ∨ s.H=0 ∨ s.T=0 ∨
    (s.B*s.H≤65535 ∧ blocks s.T 64≤65535 ∧ Machine.formedLimit s<2^31))

def PointerRoom (c : Call) : Prop := Machine.Empty c.shape ∨
  (c.q.pointer+4*Machine.formedLimit c.shape<2^63 ∧
   c.k.pointer+4*Machine.formedLimit c.shape<2^63 ∧
   c.v.pointer+4*Machine.formedLimit c.shape<2^63)
instance (c : Call) : Decidable (PointerRoom c) := by unfold PointerRoom Machine.Empty;infer_instance

/-- The redundant explicit stride check records the ordinary contiguous
Tensor guarantee while permitting every irrelevant singleton stride. -/
def DenseStrides (s : Shape) (width : Nat) (t : TensorView) : Prop :=
  if s.B=0 ∨ s.H=0 ∨ s.T=0 then True else
  match t.strides with
  | [sb,sh,st,se] => (s.B≤1 ∨ sb=(s.H*s.T*width : Nat)) ∧
      (s.H≤1 ∨ sh=(s.T*width : Nat)) ∧ (s.T≤1 ∨ st=(width : Int)) ∧ (width≤1 ∨ se=1)
  | _ => False
instance (s : Shape) (width : Nat) (t : TensorView) : Decidable (DenseStrides s width t) := by
  unfold DenseStrides
  split
  · infer_instance
  · split <;> infer_instance

structure Domain (c : Call) : Prop where
  shape : ShapeContract c.shape
  qKind : TensorKind c.q
  kKind : TensorKind c.k
  vKind : TensorKind c.v
  metadata : MetadataBinding c
  strides : DenseStrides c.shape c.shape.K c.q ∧ DenseStrides c.shape c.shape.K c.k ∧ DenseStrides c.shape c.shape.V c.v
  devices : c.q.tensor.device=c.k.tensor.device ∧ c.q.tensor.device=c.v.tensor.device
  qStorage : Stored c.q (c.shape.B*c.shape.H*c.shape.T*c.shape.K)
  kStorage : Stored c.k (c.shape.B*c.shape.H*c.shape.T*c.shape.K)
  vStorage : Stored c.v (c.shape.B*c.shape.H*c.shape.T*c.shape.V)
  admission : Machine.Admission c.shape
  room : PointerRoom c

inductive Error where
  | object | scalar | metadata | shape | device | storage | launch | pointer
  deriving DecidableEq

def checkCall (c : Call) : Except Error Unit :=
  if ¬TensorKind c.q ∨ ¬TensorKind c.k ∨ ¬TensorKind c.v then .error .metadata
  else if shapeCheck c.shape≠true ∨ ¬MetadataBinding c then .error .shape
  else if ¬DenseStrides c.shape c.shape.K c.q ∨ ¬DenseStrides c.shape c.shape.K c.k ∨
      ¬DenseStrides c.shape c.shape.V c.v then .error .storage
  else if c.q.tensor.device≠c.k.tensor.device ∨ c.q.tensor.device≠c.v.tensor.device then .error .device
  else if ¬Stored c.q (c.shape.B*c.shape.H*c.shape.T*c.shape.K) ∨
      ¬Stored c.k (c.shape.B*c.shape.H*c.shape.T*c.shape.K) ∨
      ¬Stored c.v (c.shape.B*c.shape.H*c.shape.T*c.shape.V) then .error .storage
  else if machineCheck c.shape≠true then .error .launch
  else if ¬PointerRoom c then .error .pointer else .ok ()

/-- Python object classification precedes querying Tensor metadata. Flags are
strict bools; eps is an int/float (excluding bool) convertible to Python float.
All IEEE epsilon values are admitted, including infinities and NaNs. -/
inductive ClassifiedCall where
  | nonTensor | otherScalar | floatOverflow | query (call : Call)

def check : ClassifiedCall → Except Error Unit
  | .nonTensor => .error .object
  | .otherScalar | .floatOverflow => .error .scalar
  | .query c => checkCall c

def Admitted : ClassifiedCall → Prop
  | .query c => Domain c
  | _ => False

/-- Word-indexed semantic handle, tied to the actual aligned byte pointer. -/
def inputs (c : Call) : HostAPI.Inputs :=
  ⟨HostAPI.tensor c.shape .query (c.q.pointer/4),
   HostAPI.tensor c.shape .key (c.k.pointer/4),
   HostAPI.tensor c.shape .value (c.v.pointer/4)⟩

end VeriTile.Bench.Repairs.ParallelAttention.Checked
