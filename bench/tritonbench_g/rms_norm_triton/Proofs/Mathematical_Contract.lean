import VeriTile.Torch.Tensor
import Mathlib.Analysis.SpecialFunctions.Sqrt
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

namespace Whole
open scoped BigOperators
noncomputable section

/-- Independent logical RMSNorm. The sum ranges over exactly N logical input
entries and does not mention memory, padded tiles, the DSL or its interpreter. -/
def realFormula (N : Nat) (eps : ℝ) (xs ws : Nat → ℝ) (i : Nat) : ℝ :=
  xs i * (1 / Real.sqrt ((∑ j ∈ Finset.range N, xs j * xs j) / (N : ℝ) + eps)) * ws i

theorem realFormula_div (N : Nat) (eps : ℝ) (xs ws : Nat → ℝ) (i : Nat) :
    realFormula N eps xs ws i =
      (xs i / Real.sqrt ((∑ j ∈ Finset.range N, xs j*xs j)/(N : ℝ)+eps))*ws i := by
  simp [realFormula,div_eq_mul_inv]

def RegularRow (N : Nat) (eps : ℝ) (xs : Nat → ℝ) : Prop :=
  0 < N ∧ 0 < (∑ j ∈ Finset.range N, xs j * xs j) / (N : ℝ) + eps

abbrev Memory (α : Type) := Nat → Nat → α

structure Tensor where
  shape : List Nat
  allocation : Nat
  base : Nat
  offset : Nat → Nat
  dtype : Storage
  device : Nat

def Tensor.numel (x : Tensor) := x.shape.prod
def Tensor.read (x : Tensor) (mem : Memory α) (i : Nat) : α := mem x.allocation (x.offset i)
def Contiguous (x : Tensor) : Prop := ∀ i, i < x.numel → x.offset i = x.base+i

def dim (input : Tensor) (normalized : List Nat) := input.shape.length-normalized.length
def rows (input : Tensor) (normalized : List Nat) := (input.shape.take (dim input normalized)).prod
def columns (normalized : List Nat) := normalized.prod

/-- Standard RMSNorm shape/precision premises. They are public conditions, not
invented Python assertions. Noncontiguous logical inputs and shared X/W storage
are allowed; the source explicitly makes its two contiguous copies. -/
structure InputDomain (input weight : Tensor) (normalized : List Nat)
    (preexisting : Nat → Prop) : Prop where
  rank : normalized.length ≤ input.shape.length
  suffix : input.shape.drop (dim input normalized) = normalized
  weightShape : weight.shape = normalized
  inputDType : input.dtype = .fp32
  weightDType : weight.dtype = .fp32
  sameDevice : input.device = weight.device
  inputExisted : preexisting input.allocation
  weightExisted : preexisting weight.allocation

/-- Ordinary contiguous-copy contract, including its possible alias of an
already-contiguous input. It supplies copied input values, never RMSNorm values. -/
structure Copied (α : Type) (input : Tensor) (entry : Memory α) (preexisting : Nat → Prop) where
  tensor : Tensor
  memory : Memory α
  shape : tensor.shape = input.shape
  dtype : tensor.dtype = input.dtype
  device : tensor.device = input.device
  contiguous : Contiguous tensor
  values : ∀ i, i < input.numel → tensor.read memory i = input.read entry i
  frame : ∀ allocation offset, preexisting allocation → memory allocation offset = entry allocation offset

/-- Ordinary empty_like allocation after both copies; output bytes are initially
arbitrary, fresh and contiguous because the template input copy is contiguous. -/
structure Allocated (α : Type) (template : Tensor) (entry : Memory α) (preexisting : Nat → Prop) where
  tensor : Tensor
  memory : Memory α
  shape : tensor.shape = template.shape
  dtype : tensor.dtype = template.dtype
  device : tensor.device = template.device
  contiguous : Contiguous tensor
  fresh : ¬ preexisting tensor.allocation
  frame : ∀ allocation offset, preexisting allocation → memory allocation offset = entry allocation offset

structure Prepared (α : Type) (input weight : Tensor) (entry : Memory α) (preexisting : Nat → Prop) where
  xc : Copied α input entry preexisting
  wc : Copied α weight xc.memory (fun a => preexisting a ∨ a = xc.tensor.allocation)
  output : Allocated α xc.tensor wc.memory
    (fun a => preexisting a ∨ a = xc.tensor.allocation ∨ a = wc.tensor.allocation)


/-- Row-major logical indexing through the actual queried strides. Zero strides
admit broadcast views; a nonzero storage offset admits slices. -/
def stridedOffset : List Nat → List Nat → Nat → Nat
  | [], _, _ => 0
  | _::shape, stride::strides, i => (i / shape.prod)*stride +
      stridedOffset shape strides (i % shape.prod)
  | _, [], _ => 0

/-- PyTorch API observations for an actual view and its backing allocation.
The pointer includes the storage offset. The proof does not require original
inputs to be contiguous: the Python source intentionally copies both inputs. -/
structure TensorView where
  metadata : VeriTile.Torch.TensorMetadata
  strides : List Nat
  allocation : Nat
  storageBase : Nat
  storageOffset : Nat
  capacity : Nat

def TensorView.numel (v : TensorView) := v.metadata.shape.prod
def TensorView.offset (v : TensorView) (i : Nat) :=
  v.storageOffset + stridedOffset v.metadata.shape v.strides i
def TensorView.read (v : TensorView) (mem : Memory α) (i : Nat) :=
  mem v.allocation (v.offset i)
def TensorView.address (v : TensorView) (i : Nat) :=
  v.storageBase + 4*v.offset i

/-- Bind a semantic Tensor to the same actual Tensor's metadata and physical
logical-index map. These are ordinary API observations, never output values. -/
structure TensorBinding (t : Tensor) (v : TensorView) : Prop where
  shape : t.shape = v.metadata.shape
  allocation : t.allocation = v.allocation
  base : t.base = v.storageOffset
  offsets : ∀ i, i < t.numel → t.offset i = v.offset i
  rank : v.strides.length = v.metadata.shape.length
  dtype : t.dtype = .fp32
  fp32 : v.metadata.dtype = .floating .fp32
  layout : v.metadata.layout = .strided
  device : t.device = v.metadata.device.index
  cuda : v.metadata.device.kind = "cuda"
  pointer : 0 < v.numel → v.metadata.dataPtr.toNat = v.storageBase+4*v.storageOffset
  capacity : ∀ i, i < v.numel → v.offset i < v.capacity
  addressRange : v.storageBase+4*v.capacity ≤ 2^64

/-- All source arguments, the two actual contiguous-copy results, and the
actual output allocation are bound. No runtime/allocation implementation is
part of the kernel theorem. `Copied.values` is only the ordinary copy law. -/
structure TensorBindings (a : Prepared α input weight entry preexisting)
    (x w xc wc y : TensorView) : Prop where
  input : TensorBinding input x
  weight : TensorBinding weight w
  inputCopy : TensorBinding a.xc.tensor xc
  weightCopy : TensorBinding a.wc.tensor wc
  output : TensorBinding a.output.tensor y

/-- Kernel observable independent of the source, DSL and interpreter. The
frame quantifies over every physical allocation cell, including masked tails. -/
def KernelResult (formula : Nat → α) (output : TensorView)
    (initial final : Memory α) : Prop :=
  (∀ i, i < output.numel → output.read final i = formula i) ∧
  (∀ allocation offset,
    (∀ i, i < output.numel → (allocation,offset) ≠ (output.allocation,output.offset i)) →
      final allocation offset = initial allocation offset)

/-- Source pointer expressions for the actual copies and output allocation. -/
def KernelPointers (xc wc y : TensorView) : Prop :=
  (∀ i, i < xc.numel → xc.address i = xc.metadata.dataPtr.toNat+4*i) ∧
  (∀ i, i < wc.numel → wc.address i = wc.metadata.dataPtr.toNat+4*i) ∧
  (∀ i, i < y.numel → y.address i = y.metadata.dataPtr.toNat+4*i)


/-- Independent whole-memory update, useful for stating equality of every
physical cell rather than only the active outputs. -/
def referenceMemory (formula : Nat → α) (output : TensorView)
    (initial : Memory α) : Memory α := fun allocation offset =>
  if allocation = output.allocation ∧ output.storageOffset ≤ offset ∧
      offset < output.storageOffset+output.numel then formula (offset-output.storageOffset)
  else initial allocation offset


end
#guard_msgs (drop info) in
#auditModuleAxioms
end Whole
end VeriTile.Bench.TritonBenchG.RmsNormTriton
