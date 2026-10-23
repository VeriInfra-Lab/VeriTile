import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.kv_cache_filling.Kernels

namespace VeriTile.Bench.Repairs.KvCacheFilling

/-- Input metadata and ownership conditions checked by the actual host, before
the first launch. No value, final state, kernel output or success occurs here. -/
structure Domain (r : Request) : Prop where
  heads_pos : 0 < r.heads
  block_pos : 0 < r.block
  dk_pos : 0 < r.dk
  dv_pos : 0 < r.dv
  key_shape : r.dk = r.factor * r.ck
  value_shape : r.dv = r.factor * r.cv
  lengths : ∀ b, b < r.batches → r.count b ≤ r.maxQ ∧
    r.start b + r.count b ≤ r.tokens ∧ r.count b ≤ r.total b
  page_bounds : ∀ b t, b < r.batches → t < r.count b →
    (r.history b + t) / r.block < r.tableWidth ∧
    r.page b ((r.history b + t) / r.block) < r.pages
  token_ownership : ∀ b t b' t', b < r.batches → t < r.count b →
    b' < r.batches → t' < r.count b' →
    destinationToken r b t = destinationToken r b' t' → b = b' ∧ t = t'

/-- Finite metadata checks, independent of kernel execution or payloads.
The unique-destination clause is equivalent to the Python occupied-set loop. -/
def checkDomain (r : Request) : Bool :=
  decide (0<r.heads) && decide (0<r.block) && decide (0<r.dk) && decide (0<r.dv) &&
  decide (r.dk=r.factor*r.ck) && decide (r.dv=r.factor*r.cv) &&
  decide (∀ b : Fin r.batches, r.count b ≤ r.maxQ ∧
    r.start b+r.count b ≤ r.tokens ∧ r.count b ≤ r.total b) &&
  decide (∀ b : Fin r.batches, ∀ t : Fin (r.count b),
    (r.history b+t.val)/r.block<r.tableWidth ∧
    r.page b ((r.history b+t.val)/r.block)<r.pages) &&
  decide (∀ b : Fin r.batches, ∀ t : Fin (r.count b),
    ∀ b' : Fin r.batches, ∀ t' : Fin (r.count b'),
    destinationToken r b t=destinationToken r b' t' → b.val=b'.val ∧ t.val=t'.val)

/-- Actual queried metadata and backing allocation span. No logical value
or kernel result is stored in this descriptor. -/
structure TensorDescriptor where
  metadata : VeriTile.Torch.TensorMetadata
  storageBase : Nat
  storageBytes : Nat
  conjugate : Bool
  deriving DecidableEq

structure TensorInputs where
  (key value keyCache valueCache start count total table : TensorDescriptor)
  (keyMeta valueMeta : Option TensorDescriptor)

namespace TensorInputs

def dataRoles (t : TensorInputs) (p : Policy) : List TensorDescriptor :=
  [t.key,t.value,t.keyCache,t.valueCache] ++
    (if p=.copy then [] else t.keyMeta.toList++t.valueMeta.toList)

def readRoles (t : TensorInputs) : List TensorDescriptor :=
  [t.key,t.value,t.start,t.count,t.total,t.table]

def outputRoles (t : TensorInputs) (p : Policy) : List TensorDescriptor :=
  [t.keyCache,t.valueCache] ++ (if p=.copy then [] else t.keyMeta.toList++t.valueMeta.toList)

def allRoles (t : TensorInputs) (p : Policy) : List TensorDescriptor :=
  t.readRoles ++ t.outputRoles p

def metadataShapes (t : TensorInputs) (r : Request) : Prop :=
  t.key.metadata.shape=[r.tokens,r.heads,r.dk] ∧
  t.value.metadata.shape=[r.tokens,r.heads,r.dv] ∧
  t.keyCache.metadata.shape=[r.pages,r.block,r.heads,r.ck] ∧
  t.valueCache.metadata.shape=[r.pages,r.block,r.heads,r.cv] ∧
  t.start.metadata.shape=[r.batches] ∧ t.count.metadata.shape=[r.batches] ∧
  t.total.metadata.shape=[r.batches] ∧ t.table.metadata.shape=[r.batches,r.tableWidth]

def metadataTypes (t : TensorInputs) (r : Request) : Prop :=
  t.key.metadata.dtype=.floating .fp32 ∧ t.value.metadata.dtype=.floating .fp32 ∧
  t.keyCache.metadata.dtype=(if r.policy=.copy then .floating .fp32 else .other "uint8") ∧
  t.valueCache.metadata.dtype=(if r.policy=.copy then .floating .fp32 else .other "uint8") ∧
  t.start.metadata.dtype=.other "int32" ∧ t.count.metadata.dtype=.other "int32" ∧
  t.total.metadata.dtype=.other "int32" ∧ t.table.metadata.dtype=.other "int32"

def metadataScales (t : TensorInputs) (r : Request) : Prop :=
  if r.policy=.copy then True else
  match t.keyMeta,t.valueMeta with
  | some k,some v => k.metadata.shape=[r.pages,r.block,r.heads,2] ∧
      v.metadata.shape=[r.pages,r.block,r.heads,2] ∧
      k.metadata.dtype=.floating .fp32 ∧ v.metadata.dtype=.floating .fp32
  | _,_ => False

abbrev separate (a b : TensorDescriptor) : Prop :=
  a.storageBytes=0 ∨ b.storageBytes=0 ∨
  a.storageBase+a.storageBytes ≤ b.storageBase ∨ b.storageBase+b.storageBytes ≤ a.storageBase

/-- Payload byte width after dtype admission. Metadata vectors are transported
through Torch's ordinary logical .cpu().tolist() interface. -/
def bytes (t : TensorDescriptor) : Nat := if t.metadata.dtype=.other "uint8" then 1 else 4

abbrev metadataViews (t : TensorInputs) (r : Request) : Prop :=
  (∀ x∈t.allRoles r.policy, x.metadata.layout=.strided ∧ x.metadata.device=t.key.metadata.device) ∧
  t.key.metadata.device.kind="cuda" ∧
  (∀ x∈t.dataRoles r.policy, x.metadata.isContiguous=true ∧ x.metadata.isNeg=false ∧
    x.conjugate=false ∧ (x.metadata.numel=0 ∨ x.metadata.dataPtr.toNat % bytes x=0))

abbrev metadataSeparation (t : TensorInputs) (r : Request) : Prop :=
  (t.outputRoles r.policy).Pairwise separate ∧
  (∀ a∈t.outputRoles r.policy, ∀ b∈t.readRoles, separate a b)

abbrev nativeLimits (t : TensorInputs) (r : Request) : Prop :=
  r.ck ≤ 2^20 ∧ r.cv ≤ 2^20 ∧ (jobs r).length<2^31 ∧
  (∀ x∈t.dataRoles r.policy, x.metadata.numel*bytes x<2^63)

instance (t : TensorInputs) (r : Request) : Decidable (t.metadataShapes r) :=
  inferInstanceAs (Decidable (_ ∧ _))
instance (t : TensorInputs) (r : Request) : Decidable (t.metadataTypes r) :=
  inferInstanceAs (Decidable (_ ∧ _))
instance (t : TensorInputs) (r : Request) : Decidable (t.metadataScales r) := by
  unfold metadataScales
  split_ifs <;> try infer_instance
  split <;> infer_instance

end TensorInputs

/-- Typed Tensor and signed-metadata classification has already succeeded.
No numerical or whole-result condition is part of admission. -/
def InputContract (t : TensorInputs) (r : Request) : Prop :=
  t.metadataShapes r ∧ t.metadataTypes r ∧ t.metadataScales r ∧
  t.metadataViews r ∧ t.metadataSeparation r ∧ t.nativeLimits r ∧ Domain r

def checkInputs (t : TensorInputs) (r : Request) : Bool :=
  decide (t.metadataShapes r) && decide (t.metadataTypes r) &&
  decide (t.metadataScales r) && decide (t.metadataViews r) &&
  decide (t.metadataSeparation r) && decide (t.nativeLimits r) && checkDomain r

/-- Logical signed values obtained by the four ordinary Torch metadata reads.
Inactive page entries are unrestricted. -/
structure SignedMetadata where
  (start count total : Nat → Int)
  page : Nat → Nat → Int

def SignedMetadata.request (s : SignedMetadata) (dimensions : Request) : Request :=
  { dimensions with
    start := fun b => (s.start b).toNat
    count := fun b => (s.count b).toNat
    total := fun b => (s.total b).toNat
    page := fun b p => (s.page b p).toNat }

def SignedMetadata.nonnegative (s : SignedMetadata) (r : Request) : Prop :=
  (∀ b : Fin r.batches, 0 ≤ s.start b ∧ 0 ≤ s.count b ∧ 0 ≤ s.total b) ∧
  (∀ b : Fin r.batches, ∀ i : Fin (r.count b),
    0 ≤ s.page b ((r.history b+i.val)/r.block))

instance (s : SignedMetadata) (r : Request) : Decidable (s.nonnegative r) :=
  inferInstanceAs (Decidable (_ ∧ _))

def Admission (t : TensorInputs) (dimensions : Request) (s : SignedMetadata) : Prop :=
  InputContract t (s.request dimensions) ∧ s.nonnegative (s.request dimensions)

def checkAdmission (t : TensorInputs) (dimensions : Request) (s : SignedMetadata) : Bool :=
  checkInputs t (s.request dimensions) && decide (s.nonnegative (s.request dimensions))

end VeriTile.Bench.Repairs.KvCacheFilling
