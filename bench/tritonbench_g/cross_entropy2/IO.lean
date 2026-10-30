import bench.tritonbench_g.cross_entropy2.Kernels
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.ProgramIO
import bench.tritonbench_g.cross_entropy2.Proofs.KernelReal
import bench.tritonbench_g.cross_entropy1.IO

namespace VeriTile.Bench.Repairs.CrossEntropy2.IO
open VeriTile.Triton FP.Structural
noncomputable section

def forwardInputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.cols],p.logits⟩,⟨.int,[],Region.cast p.labels⟩]
def forwardOutputs (p : Parameters) : List IOChannel :=
  [⟨.real,[],p.lse⟩,⟨.real,[],p.loss⟩,⟨.real,[],p.zLoss⟩]
def backwardInputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.cols],p.logits⟩,⟨.int,[],Region.cast p.labels⟩,
    ⟨.real,[],p.lse⟩,⟨.real,[],p.upstream⟩]

/-- Runtime ABI is installed as arguments, with no replacement source literals.
The complete original source handles both block PID and signed labels. -/
def launch {α : Type} (A : Algebra α) (p : Parameters)
    (source : ComputeKernel) (row block : Nat) (args : RuntimeArgs.Values α) : Invocation α where
  kernel := source
  backend := A.withDefaultPrecision .fp32
  arguments := [⟨"smoothing",.real,[],fun _ => args.smoothing⟩,
    ⟨"lse_square_scale",.real,[],fun _ => args.squareScale⟩,
    ⟨"logit_scale",.real,[],fun _ => args.logitScale⟩,
    ⟨"total_classes",.int,[],fun _ => (p.totalClasses : Int)⟩]
  pids := fun ax => if ax=0 then row else if ax=1 then block else 0
  setPids := fun ax => decide (ax=0 ∨ ax=1)

def forward {α β : Type} (A : Algebra β) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values β) (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := forwardInputs p
  outputs := forwardOutputs p
  buffers := [p.logits,Region.cast p.labels,p.lse,p.loss,p.zLoss]
  scratch := []
  read := fun port _ => Fin.cases
    (fun j => p.logitsBase+row*p.logitsStride+j.1.val)
    (fun next => Fin.cases (fun _ => p.labelsBase+row)
      (fun impossible => Fin.elim0 impossible) next) port
  inputMask := fun _ _ _ => True
  write := fun port _ => Fin.cases
    (fun _ => p.lseBase+block*p.rows+row)
    (fun next => Fin.cases (fun _ => p.lossBase+block*p.rows+row)
      (fun next => Fin.cases (fun _ => p.zLossBase+block*p.rows+row)
        (fun impossible => Fin.elim0 impossible) next) next) port
  outputMask := fun _ _ _ => True
  scratchWrite := fun impossible => Fin.elim0 impossible
  scratchMask := fun impossible => Fin.elim0 impossible
  schedule := fun _ => [launch A p (forwardSyntax p) row block args]
  encode := encode
  observe := fun _ _ => observe

/-- Source upstream input is inactive on an ignored label. Gradient may alias
the input: no fresh-output requirement is introduced by this regional IO. -/
def backward {α β : Type} (A : Algebra β) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values β) (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := backwardInputs p
  outputs := [⟨.real,[p.block],p.gradient⟩]
  buffers := [p.logits,Region.cast p.labels,p.lse,p.upstream,p.gradient]
  scratch := []
  read := fun port _ => Fin.cases
    (fun j => p.logitsBase+row*p.logitsStride+j.1.val)
    (fun next => Fin.cases (fun _ => p.labelsBase+row)
      (fun next => Fin.cases (fun _ => p.lseBase+row)
        (fun next => Fin.cases (fun _ => p.upstreamBase+row*p.upstreamStride)
          (fun impossible => Fin.elim0 impossible) next) next) next) port
  inputMask := fun port vals _ => port.val≠3 ∨ vals (1 : Fin 4) PUnit.unit≠p.ignored
  write := fun port _ => Fin.cases
    (fun j => p.gradientBase+row*p.gradientStride+(block*p.block+j.1.val))
    (fun impossible => Fin.elim0 impossible) port
  outputMask := fun port _ => Fin.cases
    (fun j => block*p.block+j.1.val<p.cols) (fun impossible => Fin.elim0 impossible) port
  scratchWrite := fun impossible => Fin.elim0 impossible
  scratchMask := fun impossible => Fin.elim0 impossible
  schedule := fun _ => [launch A p (backwardSyntax p) row block args]
  encode := encode
  observe := fun _ _ => observe

def vector {α : Type} [Inhabited α] {N : Nat}
    (xs : Values α .real [N]) (i : Nat) : α :=
  if h : i<N then xs (⟨i,h⟩,PUnit.unit) else default

def forwardFormula {α : Type} [Inhabited α] (p : Parameters)
    (f : (Nat → α) → Int → α × α × α)
    (vals : ChannelValues α (forwardInputs p)) : ChannelValues α (forwardOutputs p) :=
  let result := f (vector (vals (0 : Fin 2))) (vals (1 : Fin 2) PUnit.unit)
  Fin.cases (fun _ => result.1)
    (fun next => Fin.cases (fun _ => result.2.1)
      (fun next => Fin.cases (fun _ => result.2.2) (fun impossible => Fin.elim0 impossible) next) next)

def backwardFormula {α : Type} [Inhabited α] (p : Parameters)
    (f : (Nat → α) → Int → α → α → Fin p.block → α)
    (vals : ChannelValues α (backwardInputs p)) : ChannelValues α [⟨.real,[p.block],p.gradient⟩] :=
  Fin.cases (fun j => f (vector (vals (0 : Fin 4))) (vals (1 : Fin 4) PUnit.unit)
    (vals (2 : Fin 4) PUnit.unit) (vals (3 : Fin 4) PUnit.unit) j.1)
    (fun impossible => Fin.elim0 impossible)

end
end VeriTile.Bench.Repairs.CrossEntropy2.IO

namespace VeriTile.Bench.Repairs.CrossEntropy2.RealIO
open VeriTile.Triton FP.Structural
noncomputable section

def decode (x : WithBot ℝ) : ℝ := x.unbotD 0

def forwardValues (p : Parameters) (xs : ChannelValues ℝ (IO.forwardInputs p)) :
    ChannelValues (WithBot ℝ) (IO.forwardInputs p) :=
  Fin.cases (fun j => some (xs (0 : Fin 2) j))
    (fun next => Fin.cases (fun j => xs (1 : Fin 2) j) (fun impossible => Fin.elim0 impossible) next)

def backwardValues (p : Parameters) (xs : ChannelValues ℝ (IO.backwardInputs p)) :
    ChannelValues (WithBot ℝ) (IO.backwardInputs p) :=
  Fin.cases (fun j => some (xs (0 : Fin 4) j))
    (fun next => Fin.cases (fun j => xs (1 : Fin 4) j)
      (fun next => Fin.cases (fun j => some (xs (2 : Fin 4) j))
        (fun next => Fin.cases (fun j => some (xs (3 : Fin 4) j))
          (fun impossible => Fin.elim0 impossible) next) next) next)

end
end VeriTile.Bench.Repairs.CrossEntropy2.RealIO

namespace VeriTile.Bench.Repairs.CrossEntropy2.LogicalIO
open VeriTile.Triton FP.Structural
open CrossEntropy1.LogicalData
noncomputable section

def forward (p : Parameters) (row block : Nat) :
    LogicalProgramIO ℝ (WithBot ℝ) (ForwardInputs ℝ p.cols) (ForwardOutputsZ ℝ) where
  core := IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some (fun x => x.unbotD 0)
  inputs := forwardInputsEquiv ℝ p.cols p.logits (Region.cast p.labels)
  outputs := forwardOutputsZEquiv ℝ p.lse p.loss p.zLoss

def backward (p : Parameters) (row block : Nat) :
    LogicalProgramIO ℝ (WithBot ℝ) (BackwardInputs ℝ p.cols) (Fin p.block → ℝ) where
  core := IO.backward KernelReal.algebra p row block (KernelReal.arguments p) some (fun x => x.unbotD 0)
  inputs := backwardInputsEquiv ℝ p.cols p.logits (Region.cast p.labels) p.lse p.upstream
  outputs := gradientEquiv ℝ p.block p.gradient

def unsplitDomain (p : Parameters) (xs : ForwardInputs ℝ p.cols) : Prop :=
  xs.label = p.ignored ∨ (0 ≤ xs.label ∧ xs.label < (p.cols : Int))

end
end VeriTile.Bench.Repairs.CrossEntropy2.LogicalIO
