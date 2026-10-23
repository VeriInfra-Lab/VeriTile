import bench.repairs.tritonbench_g.triton_argmax.Proofs.RequestBinding
import bench.repairs.tritonbench_g.triton_argmax.Proofs.RealScratch
import bench.repairs.tritonbench_g.square_matrix.Contract

namespace VeriTile.Bench.Repairs.TritonArgmax.Checked
open Source

inductive DTypeArgument where
  | omitted
  | floating : RealScratch.FloatingKind → DTypeArgument
  | integer : Formats.IntegerKind → DTypeArgument
  | invalid : Nat → DTypeArgument

def DTypeArgument.adapter : DTypeArgument → Option RealScratch.Adapter
  | .omitted => some (.directCast RealScratch.FloatingKind.fp32.direct)
  | .floating kind => some (.directCast kind.direct)
  | .integer kind => some (.integerSafe kind)
  | .invalid _ => none

/-- The host scalar is the observed result of Python ceil(sqrt(numel)).
The six-shift Triton next_power_of_2 helper is then interpreted literally. -/
structure Observation where
  input : VeriTile.Torch.TensorMetadata
  dim : Option Int
  keepdim : Bool
  dtype : DTypeArgument
  sqrtCeil : Nat
  prepared : VeriTile.Torch.TensorMetadata

def block (o : Observation) := SquareMatrix.block o.sqrtCeil
def blocks (o : Observation) := (o.input.numel + block o - 1) / block o
def midBlock (o : Observation) := SquareMatrix.block (blocks o)
def outer (o : Observation) (ax : Nat) := (o.input.shape.take ax).prod
def extent (o : Observation) (ax : Nat) := o.input.shape[ax]?.getD 0
def inner (o : Observation) (ax : Nat) := (o.input.shape.drop (ax+1)).prod

def BaseContract (o : Observation) : Prop :=
  o.input.layout = .strided ∧ o.input.dtype = .floating .fp32 ∧
  o.input.device.kind = "cuda" ∧ 4*o.input.numel < 2^63

def GlobalContract (o : Observation) : Prop :=
  o.input.isContiguous = Bool.true ∧ o.input.isNeg = Bool.false ∧
  0 < o.input.numel ∧ o.dtype.adapter.isSome = Bool.true ∧
  0 < block o ∧ block o ≤ 2^20 ∧
  0 < blocks o ∧ blocks o ≤ midBlock o ∧ midBlock o ≤ 2^20 ∧
  o.input.numel + block o < 2^63 ∧ 8*midBlock o < 2^63 ∧
  o.input.dataPtr.toNat % 4 = 0 ∧
  o.input.dataPtr.toNat + 4*o.input.numel ≤ 2^64

def DimensionContract (o : Observation) (ax : Nat) : Prop :=
  0 < extent o ax ∧ (0 < outer o ax ∧ 0 < inner o ax →
    8*(outer o ax*inner o ax) < 2^63 ∧
    outer o ax + 128 < 2^63 ∧ extent o ax + 128 < 2^63 ∧
    inner o ax < 2^63 ∧ ((outer o ax+127)/128)*inner o ax < 2^63 ∧
    o.prepared.isNeg = Bool.false ∧ o.prepared.dataPtr.toNat % 4 = 0 ∧
    o.prepared.dataPtr.toNat + 4*o.prepared.numel ≤ 2^64)

def InputContract (o : Observation) : Prop :=
  BaseContract o ∧ match o.dim with
  | none => GlobalContract o
  | some dim => match Host.normalizeDim o.input.shape.length dim with
    | none => False
    | some ax => DimensionContract o ax

instance (o : Observation) : Decidable (BaseContract o) := by unfold BaseContract; infer_instance
instance (o : Observation) : Decidable (GlobalContract o) := by unfold GlobalContract; infer_instance
instance (o : Observation) (ax : Nat) : Decidable (DimensionContract o ax) := by
  unfold DimensionContract; infer_instance

def check (o : Observation) : Bool :=
  decide (BaseContract o) && match o.dim with
  | none => decide (GlobalContract o)
  | some dim => match Host.normalizeDim o.input.shape.length dim with
    | none => Bool.false
    | some ax => decide (DimensionContract o ax)

/-- Dynamic Python argument classification precedes all tensor observations.
The dimension branch does not classify or inspect its dtype token. -/
inductive Arguments where
  | wrongTensor | wrongDim | wrongKeepdim
  | typed : Observation → Arguments

def checkArguments : Arguments → Bool
  | .typed o => check o
  | _ => Bool.false

def AdmittedArguments : Arguments → Prop
  | .typed o => InputContract o
  | _ => False

end VeriTile.Bench.Repairs.TritonArgmax.Checked
