import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.kldiv_ops.Kernels
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ProfiledRewrite

/-! Public IO data for the two selected KL kernels. Forward exposes either the
full elementwise output or one accumulated scalar per row. Backward exposes the
full prediction derivative. Both schedules retain every source row and chunk. -/
namespace VeriTile.Bench.Repairs.KldivOps.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def outputLength (p : Parameters) (red : Nat) : Nat := if red=0 then p.rows*p.cols else p.rows
def forwardInputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.rows*p.cols],p.Y⟩,⟨.real,[p.rows*p.cols],p.T⟩]
def backwardInputs (p : Parameters) : List IOChannel := [⟨.real,[p.rows*p.cols],p.T⟩]
def outputChannels (r : RegionName) (n : Nat) : List IOChannel := [⟨.real,[n],r⟩]
abbrev ForwardValues (α : Type) (p : Parameters) := ChannelValues α (forwardInputs p)
abbrev BackwardValues (α : Type) (p : Parameters) := ChannelValues α (backwardInputs p)
abbrev Outputs (α : Type) (r : RegionName) (n : Nat) := ChannelValues α (outputChannels r n)

def forwardIndex (p : Parameters) : (i : Fin (forwardInputs p).length) →
    TileIndex (forwardInputs p)[i].shape → Fin (p.rows*p.cols)
  | ⟨0,_⟩ => fun j => j.1
  | ⟨1,_⟩ => fun j => j.1
  | ⟨i+2,h⟩ => False.elim (by simp [forwardInputs] at h)
def backwardIndex (p : Parameters) : (i : Fin (backwardInputs p).length) →
    TileIndex (backwardInputs p)[i].shape → Fin (p.rows*p.cols)
  | ⟨0,_⟩ => fun j => j.1
  | ⟨i+1,h⟩ => False.elim (by simp [backwardInputs] at h)
def outputIndex (r : RegionName) (n : Nat) : (o : Fin (outputChannels r n).length) →
    TileIndex (outputChannels r n)[o].shape → Fin n
  | ⟨0,_⟩ => fun j => j.1
  | ⟨i+1,h⟩ => False.elim (by simp [outputChannels] at h)

def forwardMemory (α : Type) (p : Parameters) (red : Nat) : MemoryIO α where
  inputs := forwardInputs p
  outputs := outputChannels p.O (outputLength p red)
  scratch := []
  buffers := [p.Y,p.T,p.O]
  read := fun i _ j => (forwardIndex p i j).val
  inputMask := fun _ _ _ => True
  write := fun o _ j => (outputIndex p.O (outputLength p red) o j).val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def backwardMemory (α : Type) (p : Parameters) : MemoryIO α where
  inputs := backwardInputs p
  outputs := outputChannels p.O (p.rows*p.cols)
  scratch := []
  buffers := [p.T,p.O]
  read := fun i _ j => (backwardIndex p i j).val
  inputMask := fun _ _ _ => True
  write := fun o _ j => (outputIndex p.O (p.rows*p.cols) o j).val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Epsilon is the runtime FP32 register, with its source cast retained. -/
def forwardInvocation {β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (epsilon : β) (row : Nat) : Invocation β where
  kernel := KernelDSL.forward p flag red
  backend := engine M
  arguments := [⟨"eps",.real,[],fun _ => epsilon⟩]
  pids := fun ax => if ax=0 then row else 0

def backwardInvocation {β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (row : Nat) : Invocation β where
  kernel := KernelDSL.backward p flag
  backend := engine M
  pids := fun ax => if ax=0 then row else 0

def forwardIO {α β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (epsilon : β) (encode : α → β) (observe : β → α) : ProgramIO α β :=
  { forwardMemory α p red with
    schedule := fun _ => (List.range p.rows).map (forwardInvocation M p flag red epsilon)
    encode := encode
    observe := fun _ _ => observe }

def backwardIO {α β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (encode : α → β) (observe : β → α) : ProgramIO α β :=
  { backwardMemory α p with
    schedule := fun _ => (List.range p.rows).map (backwardInvocation M p flag)
    encode := encode
    observe := fun _ _ => observe }

/-- The existing candidate changes only the signed output-row pointer product. -/
def forwardCandidateInvocation {β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (epsilon : β) (row : Nat) : Invocation β :=
  { forwardInvocation M p flag red epsilon row with kernel := KernelDSL.forwardCandidate p flag red }

def backwardCandidateInvocation {β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (row : Nat) : Invocation β :=
  { backwardInvocation M p flag row with kernel := KernelDSL.backwardCandidate p flag }

def forwardCandidate {α β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (epsilon : β) (encode : α → β) (observe : β → α) :
    ProgramIO.Variant (forwardIO M p flag red epsilon encode observe) where
  schedule := fun _ => (List.range p.rows).map (forwardCandidateInvocation M p flag red epsilon)
  observe := fun _ _ => observe

def backwardCandidate {α β : Type} (M : Algebra β) (p : Parameters) (flag : Bool)
    (encode : α → β) (observe : β → α) : ProgramIO.Variant (backwardIO M p flag encode observe) where
  schedule := fun _ => (List.range p.rows).map (backwardCandidateInvocation M p flag)
  observe := fun _ _ => observe

def prediction {α : Type} {p : Parameters} (values : ForwardValues α p) (i : Fin (p.rows*p.cols)) : α :=
  values ⟨0,by simp [forwardInputs]⟩ (i,PUnit.unit)
def target {α : Type} {p : Parameters} (values : ForwardValues α p) (i : Fin (p.rows*p.cols)) : α :=
  values ⟨1,by simp [forwardInputs]⟩ (i,PUnit.unit)
def backwardTarget {α : Type} {p : Parameters} (values : BackwardValues α p) (i : Fin (p.rows*p.cols)) : α :=
  values ⟨0,by simp [backwardInputs]⟩ (i,PUnit.unit)
def predictionAt {α : Type} [Inhabited α] {p : Parameters} (values : ForwardValues α p) (i : Nat) : α :=
  if h : i<p.rows*p.cols then prediction values ⟨i,h⟩ else default
def targetAt {α : Type} [Inhabited α] {p : Parameters} (values : ForwardValues α p) (i : Nat) : α :=
  if h : i<p.rows*p.cols then target values ⟨i,h⟩ else default
def backwardTargetAt {α : Type} [Inhabited α] {p : Parameters} (values : BackwardValues α p) (i : Nat) : α :=
  if h : i<p.rows*p.cols then backwardTarget values ⟨i,h⟩ else default

def result {α : Type} {r : RegionName} {n : Nat} (values : Fin n → α) : Outputs α r n
  | ⟨0,_⟩ => fun j => values j.1
  | ⟨i+1,h⟩ => False.elim (by simp [outputChannels] at h)
end
end VeriTile.Bench.Repairs.KldivOps.Interface

namespace VeriTile.Bench.Repairs.KldivOps.Interface
open VeriTile Triton FP.Structural
noncomputable section
structure LogicalForwardInputs (α : Type) (p : Parameters) where
  prediction : Fin p.rows → Fin p.cols → α
  target : Fin p.rows → Fin p.cols → α

def logicalForwardInputs (α : Type) (p : Parameters) : ForwardValues α p ≃ LogicalForwardInputs α p where
  toFun xs := ⟨fun r c => prediction xs (finProdFinEquiv (r,c)),fun r c => target xs (finProdFinEquiv (r,c))⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.prediction (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
    | ⟨1,_⟩ => fun j => xs.target (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; rcases j with ⟨i,⟨⟩⟩
      exact congrArg (fun k => xs ⟨0,by simp [forwardInputs]⟩ (k,PUnit.unit)) (finProdFinEquiv.apply_symm_apply i)
    · funext j; rcases j with ⟨i,⟨⟩⟩
      exact congrArg (fun k => xs ⟨1,by simp [forwardInputs]⟩ (k,PUnit.unit)) (finProdFinEquiv.apply_symm_apply i)
  right_inv := by intro xs; cases xs; simp [prediction,target]

def logicalBackwardInputs (α : Type) (p : Parameters) : BackwardValues α p ≃ (Fin p.rows → Fin p.cols → α) where
  toFun xs := fun r c => backwardTarget xs (finProdFinEquiv (r,c))
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs; funext o; fin_cases o; funext j; rcases j with ⟨i,⟨⟩⟩
    exact congrArg (fun k => xs ⟨0,by simp [backwardInputs]⟩ (k,PUnit.unit)) (finProdFinEquiv.apply_symm_apply i)
  right_inv := by intro xs; funext r c; simp [backwardTarget]

def outputColumns (p : Parameters) (red : Nat) : Nat := if red=0 then p.cols else 1

def logicalOutputIndex (p : Parameters) (red : Nat) : Fin (outputLength p red) ≃ (Fin p.rows × Fin (outputColumns p red)) :=
  (finCongr (show outputLength p red = p.rows*outputColumns p red by
    unfold outputLength outputColumns; split <;> simp_all)).trans finProdFinEquiv.symm

def logicalForwardOutputs (α : Type) (p : Parameters) (red : Nat) : Outputs α p.O (outputLength p red) ≃ (Fin p.rows → Fin (outputColumns p red) → α) where
  toFun xs := fun r c => xs ⟨0,by simp [outputChannels]⟩ ((logicalOutputIndex p red).symm (r,c),())
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs ((logicalOutputIndex p red) j.1).1 ((logicalOutputIndex p red) j.1).2
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨i,⟨⟩⟩; simp
  right_inv := by intro xs; funext r c; simp

def logicalBackwardOutputs (α : Type) (p : Parameters) : Outputs α p.O (p.rows*p.cols) ≃ (Fin p.rows → Fin p.cols → α) where
  toFun xs := fun r c => xs ⟨0,by simp [outputChannels]⟩ (finProdFinEquiv (r,c),())
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs; funext o; fin_cases o; funext j; rcases j with ⟨i,⟨⟩⟩
    exact congrArg (fun k => xs ⟨0,by simp [outputChannels]⟩ (k,PUnit.unit)) (finProdFinEquiv.apply_symm_apply i)
  right_inv := by intro xs; funext r c; simp
end
end VeriTile.Bench.Repairs.KldivOps.Interface

namespace VeriTile.Bench.Repairs.KldivOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section
def logicalForwardIO (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (epsilon : α) : LogicalProgramIO α α (LogicalForwardInputs α p) (Fin p.rows → Fin (outputColumns p red) → α) where
  core := forwardIO M p flag red epsilon id id
  inputs := logicalForwardInputs α p
  outputs := logicalForwardOutputs α p red

def logicalBackwardIO (M : Algebra α) (p : Parameters) (flag : Bool) : LogicalProgramIO α α (Fin p.rows → Fin p.cols → α) (Fin p.rows → Fin p.cols → α) where
  core := backwardIO M p flag id id
  inputs := logicalBackwardInputs α p
  outputs := logicalBackwardOutputs α p
end
end VeriTile.Bench.Repairs.KldivOps.Interface

namespace VeriTile.Bench.Repairs.KldivOps.Interface
noncomputable section
def columnAt {rows cols : Nat} (xs : Fin rows → Fin cols → ℝ) (row : Fin rows) (col : Nat) : ℝ :=
  if h : col<cols then xs row ⟨col,h⟩ else 0
end
end VeriTile.Bench.Repairs.KldivOps.Interface
