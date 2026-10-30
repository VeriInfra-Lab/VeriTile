import VeriTile.Torch.Tensor
import bench.tritonbench_g.embedding_triton_kernel.Kernels

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel
open VeriTile.Torch

/-- Initial-coordinate embedding formula; it does not invoke an execution. -/
def target {α : Type} (zero : α) (first last : Int) (ids : Nat → Int)
    (weight : Nat → Nat → α) (r d : Nat) : α :=
  if first ≤ ids r ∧ ids r < last then weight (ids r-first).toNat d else zero

structure Request where
  (ids weight output : TensorMetadata)
  (first last : Int)
  (n rows hidden : Nat)

namespace Request

def typed (c : Request) : Prop :=
  c.ids.layout = .strided ∧ c.weight.layout = .strided ∧ c.output.layout = .strided ∧
  c.ids.dtype = .other "int32" ∧ c.weight.dtype = .floating .fp32 ∧
  c.output.dtype = .floating .fp32

def device (c : Request) : Prop :=
  c.ids.device.kind = "cuda" ∧ c.weight.device.kind = "cuda" ∧
  c.output.device.kind = "cuda" ∧ c.ids.device = c.weight.device ∧
  c.weight.device = c.output.device

def shape (c : Request) : Prop :=
  c.ids.shape = [c.n] ∧ c.weight.shape = [c.rows,c.hidden] ∧
  c.output.shape = [c.n,c.hidden]

def direct (c : Request) : Prop :=
  c.ids.isContiguous = Bool.true ∧ c.weight.isContiguous = Bool.true ∧
  c.output.isContiguous = Bool.true ∧ c.ids.isNeg = Bool.false ∧
  c.weight.isNeg = Bool.false ∧ c.output.isNeg = Bool.false

def extent (c : Request) : Prop :=
  c.ids.numel < 2^61 ∧ c.weight.numel < 2^61 ∧ c.output.numel < 2^61 ∧
  grid c.n < 2^31 ∧ c.hidden ≤ 2^20

abbrev aligned (t : TensorMetadata) : Prop := t.numel = 0 ∨ t.dataPtr.toNat % 4 = 0

def alignment (c : Request) : Prop := aligned c.ids ∧ aligned c.weight ∧ aligned c.output

abbrev disjoint (a b : TensorMetadata) : Prop :=
  a.numel = 0 ∨ b.numel = 0 ∨
  a.dataPtr.toNat + 4*a.numel ≤ b.dataPtr.toNat ∨
  b.dataPtr.toNat + 4*b.numel ≤ a.dataPtr.toNat

def separate (c : Request) : Prop := disjoint c.output c.ids ∧ disjoint c.output c.weight

def upper (c : Request) : Int := min c.last (2^31)
abbrev intersects (c : Request) : Prop := max c.first (-(2^31)) < c.upper

def vocabulary (c : Request) : Prop :=
  c.hidden = 0 ∨ ¬c.intersects ∨ c.upper-c.first ≤ (c.rows : Int)

def firstId (c : Request) : Int := if c.intersects then c.first else 0
def lastId (c : Request) : Int := if c.intersects then c.upper else 0

/-- Exact source specialization selected by the checked Python wrapper. -/
def parameters (c : Request) : Parameters :=
  ⟨"weight","ids","out",c.firstId,c.lastId,c.hidden,c.hidden,c.n,c.hidden,c.hidden.nextPowerOfTwo⟩

def regionBounds (c : Request) : VeriTile.Triton.RegionName → Nat := fun r =>
  if r = "ids" then c.ids.numel else if r = "weight" then c.weight.numel
  else if r = "out" then c.output.numel else 0

instance (c : Request) : Decidable c.typed := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.device := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.shape := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.direct := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.extent := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.alignment := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.separate := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Request) : Decidable c.vocabulary := inferInstanceAs (Decidable (_ ∨ _))
end Request

/-- Named requirements, independent of check order. Python's dynamic Tensor
and integer classifications precede this typed attribute contract. -/
def InputContract (c : Request) : Prop :=
  c.typed ∧ c.device ∧ c.shape ∧ c.direct ∧ c.extent ∧
  c.alignment ∧ c.separate ∧ c.vocabulary

/-- Attribute guard translation in Python source order. -/
def check (c : Request) : Bool :=
  decide c.typed && decide c.device && decide c.shape && decide c.direct &&
  decide c.extent && decide c.alignment && decide c.separate && decide c.vocabulary

theorem checks_iff (c : Request) : check c = Bool.true ↔ InputContract c := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq, InputContract, and_assoc]

/-- Classification has no placeholder Tensor attributes or endpoint values. -/
inductive Arguments where
  | valid (request : Request)
  | invalid

def checkArguments : Arguments → Bool
  | .valid c => check c
  | .invalid => Bool.false

def AdmittedArguments : Arguments → Prop
  | .valid c => InputContract c
  | .invalid => False

theorem argument_checks_iff (a : Arguments) : checkArguments a = Bool.true ↔ AdmittedArguments a := by
  cases a with
  | valid c => exact checks_iff c
  | invalid => simp [checkArguments, AdmittedArguments]

/-- Per-program memory meaning. Bounds describe initial storage and actual
masked accesses, independently of an execution or its outputs. Nominal input
regions are separate from output; the two input roles may alias each other.
`pid` is the actual axis-zero program ID and each program visits 64 rows. -/
structure KernelContract (p : Parameters) (pid : Nat) (ids : Nat → Int)
    (bounds : VeriTile.Triton.RegionName → Nat) : Prop where
  ids_distinct : p.ids ≠ p.output
  weight_distinct : p.weight ≠ p.output
  width_covers : p.hidden ≤ p.width
  stride_covers : p.hidden ≤ p.outputStride
  ids_signed : ∀ r,r < p.n → -(2^31) ≤ ids r ∧ ids r < 2^31
  idAccess : ∀ a,a < 64 → pid*64+a < p.n → pid*64+a < bounds p.ids
  weightAccess : ∀ a,a < 64 → pid*64+a < p.n →
    p.first ≤ ids (pid*64+a) → ids (pid*64+a) < p.last →
    ∀ d,d < p.hidden →
      (ids (pid*64+a)-p.first).toNat*p.weightStride+d < bounds p.weight
  outputAccess : ∀ a,a < 64 → pid*64+a < p.n → ∀ d,d < p.hidden →
    (pid*64+a)*p.outputStride+d < bounds p.output
end VeriTile.Bench.Repairs.EmbeddingTritonKernel
