import VeriTile.Torch.Tensor
import bench.tritonbench_g.index_select_bwd.Kernels

namespace VeriTile.Bench.Repairs.IndexSelectBwd

structure View where
  metadata : VeriTile.Torch.TensorMetadata
  strides : List Int
  (storageBase storageBytes : Nat)
  conjugate : Bool
  deriving DecidableEq

structure Request where
  (destination incoming index : View)
  (R N C : Nat)
  indices : Nat → Int

def separate (a b : View) : Prop :=
  a.storageBytes=0 ∨ b.storageBytes=0 ∨ a.storageBase+a.storageBytes ≤ b.storageBase ∨
    b.storageBase+b.storageBytes ≤ a.storageBase

def aligned (v : View) (bytes : Nat) : Prop :=
  v.metadata.numel=0 ∨ v.metadata.dataPtr.toNat%bytes=0

def Request.metadata (r : Request) : Prop :=
  r.destination.metadata.shape=[r.R,r.C] ∧ r.incoming.metadata.shape=[r.N,r.C] ∧
  r.index.metadata.shape=[r.N] ∧ r.N ≤ r.R ∧
  r.destination.strides=r.incoming.strides ∧
  r.destination.metadata.dtype=.floating .fp32 ∧ r.incoming.metadata.dtype=.floating .fp32 ∧
  r.index.metadata.dtype=.other "int64" ∧ r.destination.metadata.device.kind="cuda" ∧
  (∀ v∈[r.destination,r.incoming,r.index],v.metadata.layout=.strided ∧
    v.metadata.device=r.destination.metadata.device ∧ v.metadata.isContiguous=Bool.true ∧
    v.metadata.isNeg=Bool.false ∧ v.conjugate=Bool.false) ∧
  aligned r.destination 4 ∧ aligned r.incoming 4 ∧ aligned r.index 8 ∧
  separate r.destination r.incoming ∧ separate r.destination r.index

def Request.native (r : Request) : Prop :=
  r.R*r.C*4<2^63 ∧ r.N*r.C*4<2^63 ∧ r.N*8<2^63

def Request.validIndices (r : Request) : Prop :=
  r.C=0 ∨ ∀ i : Fin r.N,0 ≤ r.indices i ∧ r.indices i<(r.R : Int)

instance (a b : View) : Decidable (separate a b) := inferInstanceAs (Decidable (_ ∨ _))
instance (v : View) (b : Nat) : Decidable (aligned v b) := inferInstanceAs (Decidable (_ ∨ _))
instance (r : Request) : Decidable r.metadata := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Request) : Decidable r.native := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Request) : Decidable r.validIndices := by unfold Request.validIndices; infer_instance

def InputContract (r : Request) : Prop := r.metadata ∧ r.native ∧ r.validIndices
def checkInputs (r : Request) : Bool := decide r.metadata && decide r.native && decide r.validIndices

/-- One independent derivative update: the current gradient is the left operand,
and the incoming contribution is the right operand. -/
def sliceTarget {A : Type} (add : A → A → A) (current incoming : Nat → A)
    (column : Nat) : A := add (current column) (incoming column)

/-- Independent complete backward formula on signed metadata and arbitrary Real
payloads. Repeated selected rows contribute once for every incoming row. -/
def gradientTarget (r : Request) (current incoming : Nat → ℝ) (row column : Nat) : ℝ :=
  current (row*r.C+column) + ((List.finRange r.N).map fun (i : Fin r.N) =>
    if r.indices i=(row : Int) then incoming (i.val*r.C+column) else 0).sum

/-- Independent native FP graph. The initial destination is the fold seed;
ascending incoming-row order is part of the repaired algorithm. -/
def orderedGradientTarget {A : Type} (add : A → A → A) (r : Request)
    (current incoming : Nat → A) (row column : Nat) : A :=
  (List.finRange r.N).foldl (fun acc (i : Fin r.N) =>
    if r.indices i=(row : Int) then add acc (incoming (i.val*r.C+column)) else acc)
    (current (row*r.C+column))

/-- The Python wrapper classifies non-Tensor arguments before metadata checks. -/
inductive Arguments where
  | tensors (request : Request)
  | other

def AdmittedArguments : Arguments → Prop
  | .tensors r => InputContract r
  | .other => False

def checkArguments : Arguments → Bool
  | .tensors r => checkInputs r
  | .other => Bool.false

end VeriTile.Bench.Repairs.IndexSelectBwd
