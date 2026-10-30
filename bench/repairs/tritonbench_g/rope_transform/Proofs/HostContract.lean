import bench.repairs.tritonbench_g.rope_transform.Proofs.ABI

namespace VeriTile.Bench.Repairs.RopeTransform.Checked
open VeriTile Triton FP.Structural

/-- Direct observations made before transpose/contiguous. Non-Tensor arguments
are rejected by the Python entrypoint before any metadata query. -/
def payload (t : Host.Tensor) : Bool :=
  t.strided && t.cuda && t.contiguous && !t.negative && !t.conjugate &&
  decide (t.dtype="float32") && decide (t.elementBytes=4)

def backingSeparate (a b : Host.Tensor) : Prop :=
  a.allocationBase=a.allocationEnd ∨ b.allocationBase=b.allocationEnd ∨
  a.allocationEnd≤b.allocationBase ∨ b.allocationEnd≤a.allocationBase

instance (a b : Host.Tensor) : Decidable (backingSeparate a b) :=
  inferInstanceAs (Decidable (_ ∨ _ ∨ _ ∨ _))

def fits (t : Host.Tensor) : Bool :=
  decide (t.shape.prod=0 ∨ (t.allocationBase≤t.base ∧
    t.base+4*t.shape.prod≤t.allocationEnd ∧ t.allocationEnd≤2^64 ∧ t.base%4=0))

def tensors {α : Type} [Inhabited α] (call : Host.Preparation α r) : List Host.Tensor :=
  [call.input Bool.false,call.input Bool.true,call.cosine,call.sine]

def metadataCheck {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r) : Bool :=
  (tensors call).all payload && (tensors call).all fits &&
  (tensors call).all (fun t => decide (t.device=(call.input Bool.false).device)) &&
  decide ((call.input Bool.false).shape=Host.shape r Bool.false) &&
  decide ((call.input Bool.true).shape=Host.shape r Bool.true) &&
  decide (call.cosine.shape=[r.sequence,r.half]) && decide (call.sine.shape=[r.sequence,r.half]) &&
  decide ((tensors call).Pairwise backingSeparate)

/-- These finite decisions are the launch checks in the executable wrapper.
The pointer decisions run after the original preparation operations. -/
def launchCheck {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r) : Bool :=
  decide (r.batches<2^31 ∧ r.sequence<2^31 ∧ r.qHeads<2^31 ∧ r.kHeads<2^31 ∧
    r.dimension<2^31 ∧ r.paddedDimension<2^31 ∧ r.rows<2^31) &&
  decide (∀ key : Bool,r.paddedHeads key<2^31 ∧ r.stride key<2^31) &&
  decide (∀ key : Bool,r.paddedHeads key*r.width≤1048576) &&
  decide (r.rows≠0 → (∀ key : Bool,ABI.last r key<2^31) ∧ ABI.coefficientLast r<2^31) &&
  decide (r.rows≠0 → (∀ key : Bool,(call.output key).base+4*ABI.last r key+4≤2^64) ∧
    call.cosine.base+4*ABI.coefficientLast r+4≤2^64 ∧
    call.sine.base+4*ABI.coefficientLast r+4≤2^64)

def check {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r) : Bool :=
  metadataCheck r call && launchCheck r call

end VeriTile.Bench.Repairs.RopeTransform.Checked
