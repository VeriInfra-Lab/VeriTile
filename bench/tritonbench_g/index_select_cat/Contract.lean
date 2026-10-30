import VeriTile.Torch.Tensor
import bench.tritonbench_g.index_select_cat.Kernels

namespace VeriTile.Bench.Repairs.IndexSelectCat

structure View where
  metadata : VeriTile.Torch.TensorMetadata
  (storageBase storageBytes : Nat)
  conjugate : Bool
  deriving DecidableEq

structure Request where
  (output source index : View)
  (M N D : Nat)
  indices : Nat → Int

def separate (a b : View) : Prop :=
  a.storageBytes=0 ∨ b.storageBytes=0 ∨ a.storageBase+a.storageBytes ≤ b.storageBase ∨
    b.storageBase+b.storageBytes ≤ a.storageBase

def aligned (v : View) (bytes : Nat) : Prop :=
  v.metadata.numel=0 ∨ v.metadata.dataPtr.toNat%bytes=0

def Request.metadata (r : Request) : Prop :=
  r.source.metadata.shape=[r.M,r.D] ∧ r.output.metadata.shape=[r.N,r.D] ∧ r.index.metadata.shape=[r.N] ∧
  r.source.metadata.dtype=.floating .fp32 ∧ r.output.metadata.dtype=.floating .fp32 ∧
  r.index.metadata.dtype=.other "int64" ∧ r.source.metadata.device.kind="cuda" ∧
  (∀ v∈[r.output,r.source,r.index],v.metadata.layout=.strided ∧
    v.metadata.device=r.source.metadata.device ∧ v.metadata.isContiguous=Bool.true ∧
    v.metadata.isNeg=Bool.false ∧ v.conjugate=Bool.false) ∧
  aligned r.source 4 ∧ aligned r.output 4 ∧ aligned r.index 8 ∧
  separate r.output r.source ∧ separate r.output r.index

def Request.native (r : Request) : Prop :=
  r.M*r.D*4<2^63 ∧ r.N*r.D*4<2^63 ∧ r.N*8<2^63 ∧
  r.N<2^31 ∧ (r.D+511)/512 ≤ 65535

def Request.validIndices (r : Request) : Prop :=
  r.D=0 ∨ ∀ i : Fin r.N,0 ≤ r.indices i ∧ r.indices i<(r.M : Int)

instance (a b : View) : Decidable (separate a b) := inferInstanceAs (Decidable (_ ∨ _))
instance (v : View) (b : Nat) : Decidable (aligned v b) := inferInstanceAs (Decidable (_ ∨ _))
instance (r : Request) : Decidable r.metadata := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Request) : Decidable r.native := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Request) : Decidable r.validIndices := by unfold Request.validIndices; infer_instance

def InputContract (r : Request) : Prop := r.metadata ∧ r.native ∧ r.validIndices
def checkInputs (r : Request) : Bool := decide r.metadata && decide r.native && decide r.validIndices

def Request.row (r : Request) (i : Nat) : Nat := (r.indices i).toNat

/-- Independent concatenation of the selected initial source rows. -/
def target {α : Type} (indices : Nat → Int) (source : Nat → Nat → α) (i d : Nat) : α :=
  source (indices i).toNat d

/-- Exact source specialization chosen by the checked wrapper. -/
def Request.layout (r : Request) : Layout :=
  ⟨"output","source","index",r.N,r.D,r.D,1,1,512⟩

/-- CPU validation and the kernel's actual int64 index loads observe the same
initial index array. This is an input-memory binding, not an output premise. -/
def IndexObservation (r : Request) (loaded : Nat → Int) : Prop :=
  ∀ i,i < r.N → loaded i = r.indices i

/-- One actual program of the selected two-dimensional launch. -/
def Launched (r : Request) (pid : Nat → Nat) : Prop :=
  pid 0 < r.N ∧ pid 1 < (r.D+511)/512

inductive Arguments where
  | tensors (request : Request)
  | invalid

def checkArguments : Arguments → Bool
  | .tensors r => checkInputs r
  | .invalid => Bool.false

def AdmittedArguments : Arguments → Prop
  | .tensors r => InputContract r
  | .invalid => False

end VeriTile.Bench.Repairs.IndexSelectCat
