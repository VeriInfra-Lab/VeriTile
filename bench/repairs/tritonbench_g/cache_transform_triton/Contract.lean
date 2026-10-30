import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Bounds

namespace VeriTile.Bench.Repairs.CacheTransform

/-- Actual input metadata and the signed values read by Torch's logical
cpu()/tolist() interface. No output or execution result is present. -/
structure Request where
  (lengths cos sin : VeriTile.Torch.TensorMetadata)
  (cosConjugate sinConjugate : Bool)
  (cosRows sinRows width : Nat)
  values : List Int
  mode : Mode

def Request.naturalLengths (r : Request) := r.values.map Int.toNat
def Request.outputRows (r : Request) :=
  match r.mode with | .prompt => r.naturalLengths.sum | .decode => r.values.length
def Request.count (r : Request) := r.outputRows*r.width*4

def Request.metadata (r : Request) : Prop :=
  r.lengths.layout=.strided ∧ r.cos.layout=.strided ∧ r.sin.layout=.strided ∧
  r.lengths.dtype=.other "int32" ∧ r.cos.dtype=.floating .fp32 ∧ r.sin.dtype=.floating .fp32 ∧
  r.lengths.shape=[r.values.length] ∧ r.cos.shape=[r.cosRows,r.width] ∧ r.sin.shape=[r.sinRows,r.width] ∧
  r.cos.device.kind="cuda" ∧ r.lengths.device=r.cos.device ∧ r.sin.device=r.cos.device ∧
  r.cos.isContiguous=Bool.true ∧ r.sin.isContiguous=Bool.true ∧
  r.cos.isNeg=Bool.false ∧ r.sin.isNeg=Bool.false ∧
  r.cosConjugate=Bool.false ∧ r.sinConjugate=Bool.false

def Request.lengthBounds (r : Request) : Prop :=
  ∀ v∈r.values, (if r.mode=.prompt then 0 else 1) ≤ v ∧
    v ≤ (r.cosRows : Int) ∧ v ≤ (r.sinRows : Int)

def Request.nativeBounds (r : Request) : Prop :=
  r.cosRows*r.width*4<2^63 ∧ r.sinRows*r.width*4<2^63 ∧
  r.count<2^63 ∧ blocks r.count<2^31 ∧ (0<r.count → r.outputRows*8<2^63)

instance (r : Request) : Decidable r.metadata := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Request) : Decidable r.lengthBounds := by unfold Request.lengthBounds; infer_instance
instance (r : Request) : Decidable r.nativeBounds := inferInstanceAs (Decidable (_ ∧ _))

def InputContract (r : Request) : Prop := r.metadata ∧ r.lengthBounds ∧ r.nativeBounds
def checkInputs (r : Request) : Bool := decide r.metadata && decide r.lengthBounds && decide r.nativeBounds

end VeriTile.Bench.Repairs.CacheTransform
