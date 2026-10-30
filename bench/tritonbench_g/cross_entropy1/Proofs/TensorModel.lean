import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Independent logical tensor and ordinary tensor-library interface.
Coordinates are row-major logical indices, mapped through shape, strides and
storage offset. Zero strides and overlapping views are allowed. Library resolve
and clone preserve logical values; they contain no kernel execution/result law. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.TensorModel
open VeriTile.Triton FP.Structural
noncomputable section

/-- Row-major coordinate enumeration into an arbitrary nonnegative-stride
Torch storage view. The empty shape is the scalar coordinate. -/
def coordinateOffset : List Nat → List Nat → Nat → Nat
  | [], _, _ => 0
  | d :: ds, stride :: strides, i =>
      (i / ds.prod) % d * stride + coordinateOffset ds strides (i % ds.prod)
  | _ :: _, [], _ => 0

structure Tensor where
  objectId : Nat
  region : RegionName
  shape : List Nat
  strides : List Nat
  storageOffset : Nat
  isNeg : Bool
  isCUDA : Bool
  device : Nat
  dtype : String

def Tensor.numel (t : Tensor) : Nat := t.shape.prod
def Tensor.index (t : Tensor) (i : Nat) : Nat :=
  t.storageOffset + coordinateOffset t.shape t.strides i

def Tensor.Dense (t : Tensor) : Prop :=
  t.isNeg = Bool.false ∧ ∀ i, i < t.numel → t.index i = i

def value {α : Type} [Inhabited α] (neg : α → α) (s : State α) (t : Tensor) (i : Nat) : α :=
  let raw := (s.mem t.region (t.index i)).read .fp32
  if t.isNeg then neg raw else raw

structure World (α σ : Type) where
  kernel : State α
  live : RegionName → Prop
  native : σ

/-- Ordinary input object metadata, not a new native-safe restriction.
The tensor library supplies backing validity for each logical coordinate. -/
structure Input {α σ : Type} (s : World α σ) (t : Tensor) : Prop where
  live : s.live t.region
  rank : t.strides.length = t.shape.length
  dtype : t.dtype = "torch.float32"

/-- Metadata preserved by resolve_neg, dense clone and empty_like. Object and
storage identity deliberately are absent: clone and output allocate new objects. -/
structure SameMetadata (a b : Tensor) : Prop where
  shape : b.shape = a.shape
  dtype : b.dtype = a.dtype
  cuda : b.isCUDA = a.isCUDA
  device : b.device = a.device

theorem SameMetadata.numel {a b : Tensor} (h : SameMetadata a b) : b.numel = a.numel := by
  simp [Tensor.numel, h.shape]

/-- Allocation/materialization preserves every previously live storage cell,
including cells outside all input views and padding. -/
structure Preserves {α σ : Type} (before after : World α σ) : Prop where
  live : ∀ r, before.live r → after.live r
  frame : ∀ r i, before.live r → after.kernel.mem r i = before.kernel.mem r i

theorem Preserves.refl {α σ : Type} (s : World α σ) : Preserves s s :=
  ⟨fun _ h => h, fun _ _ _ => rfl⟩

theorem Preserves.trans {α σ : Type} {a b c : World α σ}
    (ab : Preserves a b) (bc : Preserves b c) : Preserves a c :=
  ⟨fun r h => bc.live r (ab.live r h), fun r i h =>
    (bc.frame r i (ab.live r h)).trans (ab.frame r i h)⟩

theorem Preserves.value {α σ : Type} [Inhabited α] (neg : α → α)
    {before after : World α σ} (h : Preserves before after) (t : Tensor)
    (live : before.live t.region) (i : Nat) :
    value neg after.kernel t i = value neg before.kernel t i := by
  unfold VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.value
  rw [h.frame t.region (t.index i) live]

/-- Ordinary coordinate-preserving resolve_neg/clone contract. This relation
is independent of the addition operation and of all kernel definitions. -/
structure Materialized {α σ : Type} [Inhabited α] (neg : α → α)
    (input output : Tensor) (before after : World α σ) : Prop where
  metadata : SameMetadata input output
  frame : Preserves before after
  live : after.live output.region
  coordinates : ∀ i, i < input.numel →
    value neg after.kernel output i = value neg before.kernel input i

structure Cloned {α σ : Type} [Inhabited α] (neg : α → α)
    (input output : Tensor) (before after : World α σ) : Prop
    extends Materialized neg input output before after where
  dense : output.Dense
  fresh : ¬before.live output.region
  objectFresh : output.objectId ≠ input.objectId

/-- Ordinary fresh contiguous empty_like result; no initialized output values
are assumed. Full source computation must overwrite every visible element. -/
structure Allocated {α σ : Type} (input output : Tensor) (before after : World α σ) : Prop where
  metadata : SameMetadata input output
  frame : Preserves before after
  live : after.live output.region
  dense : output.Dense
  fresh : ¬before.live output.region
  objectFresh : output.objectId ≠ input.objectId

/-- Independent mathematical task: coordinatewise addition of the two caller
views, with equal shape and fresh output. The actual storage relation is below. -/
def logicalAdd {α : Type} [Inhabited α] (neg : α → α) (add : α → α → α)
    (initial : State α) (x y : Tensor) (i : Nat) : α :=
  add (value neg initial x i) (value neg initial y i)

def ReturnedValues {α : Type} [Inhabited α] (neg : α → α) (add : α → α → α)
    (initial final : State α) (x y output : Tensor) : Prop :=
  output.shape = x.shape ∧ output.Dense ∧
    ∀ i, i < x.numel → value neg final output i = logicalAdd neg add initial x y i

#specNonCircular logicalAdd avoiding [FP.Structural.exec, FP.Structural.run]
#specNonCircular ReturnedValues avoiding [FP.Structural.exec, FP.Structural.run]
#axiomsClean SameMetadata.numel
#axiomsClean Preserves.trans
#axiomsClean Preserves.value
end
end VeriTile.Bench.Repairs.CrossEntropy1.TensorModel
