import bench.tritonbench_g.cross_entropy1.Kernels
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Tactic.FinCases
import bench.tritonbench_g.cross_entropy1.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.CrossEntropy1.IO
open VeriTile.Triton FP.Structural
noncomputable section

def forwardInputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.cols],p.logits⟩,⟨.int,[],Region.cast p.labels⟩]
def forwardOutputs (p : Parameters) : List IOChannel :=
  [⟨.real,[],p.lse⟩,⟨.real,[],p.loss⟩]
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
    ⟨"total_classes",.int,[],fun _ => (p.totalClasses : Int)⟩]
  pids := fun ax => if ax=0 then row else if ax=1 then block else 0
  setPids := fun ax => decide (ax=0 ∨ ax=1)

def forward {α β : Type} (A : Algebra β) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values β) (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := forwardInputs p
  outputs := forwardOutputs p
  buffers := [p.logits,Region.cast p.labels,p.lse,p.loss]
  scratch := []
  read := fun port _ => Fin.cases
    (fun j => p.logitsBase+row*p.logitsStride+j.1.val)
    (fun next => Fin.cases (fun _ => p.labelsBase+row)
      (fun impossible => Fin.elim0 impossible) next) port
  inputMask := fun _ _ _ => True
  write := fun port _ => Fin.cases
    (fun _ => p.lseBase+block*p.rows+row)
    (fun next => Fin.cases (fun _ => p.lossBase+block*p.rows+row)
      (fun impossible => Fin.elim0 impossible) next) port
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
    (f : (Nat → α) → Int → α × α)
    (vals : ChannelValues α (forwardInputs p)) : ChannelValues α (forwardOutputs p) :=
  let result := f (vector (vals (0 : Fin 2))) (vals (1 : Fin 2) PUnit.unit)
  Fin.cases (fun _ => result.1)
    (fun next => Fin.cases (fun _ => result.2) (fun impossible => Fin.elim0 impossible) next)

def backwardFormula {α : Type} [Inhabited α] (p : Parameters)
    (f : (Nat → α) → Int → α → α → Fin p.block → α)
    (vals : ChannelValues α (backwardInputs p)) : ChannelValues α [⟨.real,[p.block],p.gradient⟩] :=
  Fin.cases (fun j => f (vector (vals (0 : Fin 4))) (vals (1 : Fin 4) PUnit.unit)
    (vals (2 : Fin 4) PUnit.unit) (vals (3 : Fin 4) PUnit.unit) j.1)
    (fun impossible => Fin.elim0 impossible)

end
end VeriTile.Bench.Repairs.CrossEntropy1.IO

namespace VeriTile.Bench.Repairs.CrossEntropy1.LogicalData
open VeriTile.Triton FP.Structural
noncomputable section

structure ForwardInputs (α : Type) (n : Nat) where
  logits : Fin n → α
  label : Int
structure BackwardInputs (α : Type) (n : Nat) extends ForwardInputs α n where
  savedLSE : α
  upstream : α
structure ForwardOutputs (α : Type) where
  lse : α
  loss : α
structure ForwardOutputsZ (α : Type) extends ForwardOutputs α where
  zLoss : α

def sample [Inhabited α] (xs : Fin n → α) (i : Nat) : α :=
  if h : i < n then xs ⟨i,h⟩ else default

def forwardInputsEquiv (α : Type) (n : Nat) (logits label : RegionName) :
    ChannelValues α [⟨.real,[n],logits⟩,⟨.int,[],label⟩] ≃ ForwardInputs α n where
  toFun := fun xs => ⟨fun i => xs 0 (i,()), xs 1 ()⟩
  invFun := fun xs => Fin.cases (fun i => xs.logits i.1)
    (Fin.cases (fun _ => xs.label) (fun o => Fin.elim0 o))
  left_inv := by
    intro xs; funext o j; fin_cases o
    · rcases j with ⟨i,u⟩; cases u; rfl
    · cases j; rfl
  right_inv := by intro xs; cases xs; rfl

def backwardInputsEquiv (α : Type) (n : Nat) (logits label lse upstream : RegionName) :
    ChannelValues α [⟨.real,[n],logits⟩,⟨.int,[],label⟩,⟨.real,[],lse⟩,⟨.real,[],upstream⟩] ≃
      BackwardInputs α n where
  toFun := fun xs => ⟨⟨fun i => xs 0 (i,()), xs 1 ()⟩,xs 2 (),xs 3 ()⟩
  invFun := fun xs => Fin.cases (fun i => xs.logits i.1)
    (Fin.cases (fun _ => xs.label) (Fin.cases (fun _ => xs.savedLSE)
      (Fin.cases (fun _ => xs.upstream) (fun o => Fin.elim0 o))))
  left_inv := by
    intro xs; funext o j; fin_cases o
    · rcases j with ⟨i,u⟩; cases u; rfl
    all_goals cases j; rfl
  right_inv := by intro xs; cases xs with | mk f l u => cases f; rfl

def forwardOutputsEquiv (α : Type) (lse loss : RegionName) :
    ChannelValues α [⟨.real,[],lse⟩,⟨.real,[],loss⟩] ≃ ForwardOutputs α where
  toFun := fun ys => ⟨ys 0 (),ys 1 ()⟩
  invFun := fun ys => Fin.cases (fun _ => ys.lse)
    (Fin.cases (fun _ => ys.loss) (fun o => Fin.elim0 o))
  left_inv := by intro ys; funext o j; fin_cases o <;> cases j <;> rfl
  right_inv := by intro ys; cases ys; rfl

def forwardOutputsZEquiv (α : Type) (lse loss zloss : RegionName) :
    ChannelValues α [⟨.real,[],lse⟩,⟨.real,[],loss⟩,⟨.real,[],zloss⟩] ≃ ForwardOutputsZ α where
  toFun := fun ys => ⟨⟨ys 0 (),ys 1 ()⟩,ys 2 ()⟩
  invFun := fun ys => Fin.cases (fun _ => ys.lse) (Fin.cases (fun _ => ys.loss)
    (Fin.cases (fun _ => ys.zLoss) (fun o => Fin.elim0 o)))
  left_inv := by intro ys; funext o j; fin_cases o <;> cases j <;> rfl
  right_inv := by intro ys; cases ys with | mk f z => cases f; rfl

def gradientEquiv (α : Type) (n : Nat) (gradient : RegionName) :
    ChannelValues α [⟨.real,[n],gradient⟩] ≃ (Fin n → α) where
  toFun := fun ys i => ys 0 (i,())
  invFun := fun ys => Fin.cases (fun j => ys j.1) (fun o => Fin.elim0 o)
  left_inv := by
    intro ys; funext o j; fin_cases o; rcases j with ⟨i,u⟩; cases u; rfl
  right_inv := by intro ys; rfl

def forwardTarget [Inhabited α] (f : (Nat → α) → Int → α × α)
    (xs : ForwardInputs α n) : ForwardOutputs α :=
  let r := f (sample xs.logits) xs.label
  ⟨r.1,r.2⟩
def forwardTargetZ [Inhabited α] (f : (Nat → α) → Int → α × α × α)
    (xs : ForwardInputs α n) : ForwardOutputsZ α :=
  let r := f (sample xs.logits) xs.label
  ⟨⟨r.1,r.2.1⟩,r.2.2⟩
def backwardTarget [Inhabited α] (f : (Nat → α) → Int → α → α → Fin b → α)
    (xs : BackwardInputs α n) : Fin b → α :=
  f (sample xs.logits) xs.label xs.savedLSE xs.upstream

end
end VeriTile.Bench.Repairs.CrossEntropy1.LogicalData

namespace VeriTile.Bench.Repairs.CrossEntropy1.LogicalIO
open VeriTile.Triton FP.Structural
open CrossEntropy1.LogicalData
noncomputable section

def forward (p : Parameters) (row block : Nat) :
    LogicalProgramIO ℝ (WithBot ℝ) (ForwardInputs ℝ p.cols) (ForwardOutputs ℝ) where
  core := IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some (fun x => x.unbotD 0)
  inputs := forwardInputsEquiv ℝ p.cols p.logits (Region.cast p.labels)
  outputs := forwardOutputsEquiv ℝ p.lse p.loss

def backward (p : Parameters) (row block : Nat) :
    LogicalProgramIO ℝ (WithBot ℝ) (BackwardInputs ℝ p.cols) (Fin p.block → ℝ) where
  core := IO.backward KernelReal.algebra p row block (KernelReal.arguments p) some (fun x => x.unbotD 0)
  inputs := backwardInputsEquiv ℝ p.cols p.logits (Region.cast p.labels) p.lse p.upstream
  outputs := gradientEquiv ℝ p.block p.gradient

def unsplitDomain (p : Parameters) (xs : ForwardInputs ℝ p.cols) : Prop :=
  xs.label = p.ignored ∨ (0 ≤ xs.label ∧ xs.label < (p.cols : Int))

end
end VeriTile.Bench.Repairs.CrossEntropy1.LogicalIO
