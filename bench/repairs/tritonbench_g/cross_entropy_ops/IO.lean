import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelBackward
import VeriTile.Triton.Float.ProgramIO
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelReal
import bench.repairs.tritonbench_g.cross_entropy1.IO

/-! The selected source's public channel layout. FP32 arithmetic and storage
remain explicit in the source AST; `.real` is its Structural cell carrier.
Labels retain their signed integer channel. Each interface describes exactly
one original row/block launch, including all outputs and the exterior frame. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def arguments (p : Parameters) (values : RuntimeArgs.Values A) : List (RegisterBinding A) :=
  [⟨"smoothing", .real, [], fun _ => values.smoothing⟩,
   ⟨"lse_square_scale", .real, [], fun _ => values.squareScale⟩,
   ⟨"logit_scale", .real, [], fun _ => values.logitScale⟩,
   ⟨"total_classes", .int, [], fun _ => (p.totalClasses : Int)⟩]

def invocation (M : Algebra A) (p : Parameters) (scalars : RuntimeArgs.Values A)
    (backward : Bool) (row block : Nat) : Invocation A where
  kernel := if backward then backwardSyntax p else forwardSyntax p
  backend := KernelNative.engine M
  arguments := arguments p scalars
  pids := fun ax => if ax = 0 then row else if ax = 1 then block else 0
  setPids := fun ax => decide (ax = 0 ∨ ax = 1)

namespace Forward
def inputs (p : Parameters) : List IOChannel :=
  [⟨.real, [p.cols], p.logits⟩, ⟨.int, [], Region.cast p.labels⟩]
def outputs (p : Parameters) : List IOChannel :=
  [⟨.real, [], p.lse⟩, ⟨.real, [], p.loss⟩, ⟨.real, [], p.zloss⟩]
abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputs p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputs p)

def logit (values : Inputs A p) (i : Fin p.cols) : A := values ⟨0, by simp [inputs]⟩ (i, PUnit.unit)
def label (values : Inputs A p) : Int := values ⟨1, by simp [inputs]⟩ PUnit.unit
def extend [Inhabited A] (values : Inputs A p) (i : Nat) : A :=
  if h : i < p.cols then logit values ⟨i, h⟩ else default

def read (p : Parameters) (row : Nat) (i : Fin (inputs p).length) :
    TileIndex (inputs p)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun j => p.logitsBase + row * p.logitsStride + j.1.val
  | ⟨1, _⟩ => fun _ => p.labelsBase + row
  | ⟨n + 2, h⟩ => False.elim (by simp [inputs] at h)
def write (p : Parameters) (row block : Nat) (o : Fin (outputs p).length) :
    TileIndex (outputs p)[o].shape → Nat :=
  match o with
  | ⟨0, _⟩ => fun _ => p.lseBase + block * p.rows + row
  | ⟨1, _⟩ => fun _ => p.lossBase + block * p.rows + row
  | ⟨2, _⟩ => fun _ => p.zlossBase + block * p.rows + row
  | ⟨n + 3, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (p : Parameters) (row block : Nat) : MemoryIO A where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.logits, Region.cast p.labels, p.lse, p.loss, p.zloss]
  read := fun i _ => read p row i
  inputMask := fun _ _ _ => True
  write := fun o _ => write p row block o
  outputMask := fun o _ _ => o.val ≠ 2 ∨ p.split = Bool.false
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def programIO (M : Algebra A) (p : Parameters) (scalars : RuntimeArgs.Values A)
    (row block : Nat) : ProgramIO A A :=
  { memory A p row block with
    schedule := fun _ => [invocation M p scalars Bool.false row block]
    encode := id
    observe := fun _ _ => id }

def result (p : Parameters) (lse loss zloss : A) : Outputs A p :=
  fun o => match o with
  | ⟨0, _⟩ => fun _ => lse
  | ⟨1, _⟩ => fun _ => loss
  | ⟨2, _⟩ => fun _ => zloss
  | ⟨n + 3, h⟩ => False.elim (by simp [outputs] at h)

/-- Mathematical inputs are ordinary reals embedded into the model's extended
carrier. This restates the existing Real theorem, not a finite-FP input guard. -/
def RealInputs (p : Parameters) (values : Inputs (WithBot ℝ) p) : Prop :=
  ∀ i : Fin p.cols, ∃ x : ℝ, logit values i = some x
def realInput (values : Inputs (WithBot ℝ) p) (i : Nat) : ℝ := (extend values i).getD 0
end Forward

namespace Backward
def inputs (p : Parameters) : List IOChannel :=
  [⟨.real, [p.cols], p.logits⟩, ⟨.int, [], Region.cast p.labels⟩,
   ⟨.real, [], p.lse⟩, ⟨.real, [], p.upstream⟩]
def outputs (p : Parameters) : List IOChannel := [⟨.real, [p.block], p.gradient⟩]
abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputs p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputs p)
def logit (values : Inputs A p) (i : Fin p.cols) : A := values ⟨0, by simp [inputs]⟩ (i, PUnit.unit)
def label (values : Inputs A p) : Int := values ⟨1, by simp [inputs]⟩ PUnit.unit
def lse (values : Inputs A p) : A := values ⟨2, by simp [inputs]⟩ PUnit.unit
def upstream (values : Inputs A p) : A := values ⟨3, by simp [inputs]⟩ PUnit.unit
def extend [Inhabited A] (values : Inputs A p) (i : Nat) : A :=
  if h : i < p.cols then logit values ⟨i, h⟩ else default
def read (p : Parameters) (row : Nat) (i : Fin (inputs p).length) :
    TileIndex (inputs p)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun j => p.logitsBase + row * p.logitsStride + j.1.val
  | ⟨1, _⟩ => fun _ => p.labelsBase + row
  | ⟨2, _⟩ => fun _ => p.lseBase + row
  | ⟨3, _⟩ => fun _ => p.upstreamBase + row * p.upstreamStride
  | ⟨n + 4, h⟩ => False.elim (by simp [inputs] at h)
def lane (p : Parameters) (o : Fin (outputs p).length) :
    TileIndex (outputs p)[o].shape → Fin p.block :=
  match o with
  | ⟨0, _⟩ => fun j => j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (p : Parameters) (row block : Nat) : MemoryIO A where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.logits, Region.cast p.labels, p.lse, p.upstream, p.gradient]
  read := fun i _ => read p row i
  inputMask := fun _ _ _ => True
  write := fun o _ j => p.gradientBase + row * p.gradientStride + (block * p.block + (lane p o j).val)
  outputMask := fun o _ j => block * p.block + (lane p o j).val < p.cols
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def programIO (M : Algebra A) (p : Parameters) (scalars : RuntimeArgs.Values A)
    (row block : Nat) : ProgramIO A A :=
  { memory A p row block with
    schedule := fun _ => [invocation M p scalars Bool.true row block]
    encode := id
    observe := fun _ _ => id }
def result (p : Parameters) (f : Fin p.block → A) : Outputs A p :=
  fun o => match o with
  | ⟨0, _⟩ => fun j => f j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)
def RealInputs (p : Parameters) (values : Inputs (WithBot ℝ) p) : Prop :=
  (∀ i : Fin p.cols, ∃ x : ℝ, logit values i = some x) ∧
    (∃ x : ℝ, lse values = some x) ∧ (∃ x : ℝ, upstream values = some x)
def realInput (values : Inputs (WithBot ℝ) p) (i : Nat) : ℝ := (extend values i).getD 0
end Backward

end
end VeriTile.Bench.Repairs.CrossEntropyOps.Interface

namespace VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO
open VeriTile.Triton FP.Structural Interface
open CrossEntropy1.LogicalData
noncomputable section

def forward (p : Parameters) (row block : Nat) :
    LogicalProgramIO (WithBot ℝ) (WithBot ℝ)
      (ForwardInputs (WithBot ℝ) p.cols) (ForwardOutputsZ (WithBot ℝ)) where
  core := Forward.programIO KernelReal.algebra p (KernelReal.arguments p) row block
  inputs := forwardInputsEquiv (WithBot ℝ) p.cols p.logits (Region.cast p.labels)
  outputs := forwardOutputsZEquiv (WithBot ℝ) p.lse p.loss p.zloss

def backward (p : Parameters) (row block : Nat) :
    LogicalProgramIO (WithBot ℝ) (WithBot ℝ)
      (BackwardInputs (WithBot ℝ) p.cols) (Fin p.block → WithBot ℝ) where
  core := Backward.programIO KernelReal.algebra p (KernelReal.arguments p) row block
  inputs := backwardInputsEquiv (WithBot ℝ) p.cols p.logits (Region.cast p.labels) p.lse p.upstream
  outputs := gradientEquiv (WithBot ℝ) p.block p.gradient

def ForwardReal (p : Parameters) (xs : ForwardInputs (WithBot ℝ) p.cols) : Prop :=
  ∀ i, ∃ x : ℝ, xs.logits i = some x
def BackwardReal (p : Parameters) (xs : BackwardInputs (WithBot ℝ) p.cols) : Prop :=
  (∀ i, ∃ x : ℝ, xs.logits i = some x) ∧
    (∃ x : ℝ, xs.savedLSE = some x) ∧ (∃ x : ℝ, xs.upstream = some x)

end
end VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO
