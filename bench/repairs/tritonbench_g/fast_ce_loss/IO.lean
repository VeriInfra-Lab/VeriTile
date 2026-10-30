import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelSource
import VeriTile.Triton.Float.ObservedProgramIO
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.FastCeLoss.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

def width (p : Parameters) (wholeRow : Bool) : Nat :=
  if wholeRow then p.vocab else p.block

def column (p : Parameters) (wholeRow : Bool) (chunk j : Nat) : Nat :=
  (if wholeRow then 0 else chunk*p.block)+j

def forwardInputs (p : Parameters) (wholeRow : Bool) : List IOChannel :=
  [⟨.int,[],p.labels.cast⟩,⟨.real,[width p wholeRow],p.logits⟩,⟨.real,[],p.logits⟩]
def forwardOutputs (p : Parameters) : List IOChannel :=
  [⟨.real,[],p.lse⟩,⟨.real,[],p.loss⟩]
def backwardInputs (p : Parameters) (wholeRow : Bool) : List IOChannel :=
  [⟨.int,[],p.labels.cast⟩,⟨.real,[width p wholeRow],p.logits⟩,
    ⟨.real,[],p.lse⟩,⟨.real,[],p.upstream⟩]
def backwardOutputs (p : Parameters) : List IOChannel := [⟨.real,[p.block],p.logits⟩]

def forwardLabel {α : Type} (p : Parameters) (wholeRow : Bool)
    (xs : ChannelValues α (forwardInputs p wholeRow)) : Int :=
  xs (⟨0,by decide⟩ : Fin 3) PUnit.unit

def backwardLabel {α : Type} (p : Parameters) (wholeRow : Bool)
    (xs : ChannelValues α (backwardInputs p wholeRow)) : Int :=
  xs (⟨0,by decide⟩ : Fin 4) PUnit.unit

def forwardRead {α : Type} (p : Parameters) (row chunk : Nat) (wholeRow : Bool) :
    (i : Fin (forwardInputs p wholeRow).length) →
      ChannelValues α (forwardInputs p wholeRow) → TileIndex ((forwardInputs p wholeRow)[i]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro _ _; exact row
  · intro _ j; exact row*p.logitsStride+column p wholeRow chunk j.1.val
  · intro xs _; exact row*p.logitsStride+(forwardLabel p wholeRow xs).toNat

def forwardInputMask {α : Type} (p : Parameters) (chunk : Nat) (chunked wholeRow : Bool) :
    (i : Fin (forwardInputs p wholeRow).length) →
      ChannelValues α (forwardInputs p wholeRow) → TileIndex ((forwardInputs p wholeRow)[i]).shape → Prop := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro _ _; exact True
  · intro _ j; exact column p wholeRow chunk j.1.val < p.vocab
  · intro xs _; exact (chunked = Bool.false ∨ chunk=0) ∧ forwardLabel p wholeRow xs ≠ -100

def forwardWrite {α : Type} (p : Parameters) (row chunk : Nat) (chunked wholeRow : Bool) :
    (o : Fin (forwardOutputs p).length) →
      ChannelValues α (forwardInputs p wholeRow) → TileIndex ((forwardOutputs p)[o]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0))
  · intro _ _; exact if chunked then row*p.chunks+chunk else row
  · intro _ _; exact row

def forwardOutputMask {α : Type} (p : Parameters) (chunk : Nat) (chunked wholeRow : Bool) :
    (o : Fin (forwardOutputs p).length) →
      ChannelValues α (forwardInputs p wholeRow) → TileIndex ((forwardOutputs p)[o]).shape → Prop := by
  refine Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0))
  · intro _ _; exact True
  · intro _ _; exact chunked = Bool.false ∨ chunk=0

def forwardInvocation {β : Type} (M : Algebra β) (p : Parameters)
    (row chunk : Nat) (isChunked : Bool) : Invocation β where
  kernel := if isChunked then KernelSource.chunked p else KernelSource.forward p
  backend := engine M
  pids := fun ax => if ax=0 then row else chunk
  setPids := fun ax => decide (ax=0) || (isChunked && decide (ax=1))

def forwardIO {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α)
    (p : Parameters) (row chunk : Nat) (isChunked wholeRow : Bool) : ProgramIO α β where
  inputs := forwardInputs p wholeRow
  outputs := forwardOutputs p
  scratch := []
  buffers := [p.labels.cast,p.logits,p.lse,p.loss]
  read := forwardRead p row chunk wholeRow
  inputMask := forwardInputMask p chunk isChunked wholeRow
  write := forwardWrite p row chunk isChunked wholeRow
  outputMask := forwardOutputMask p chunk isChunked wholeRow
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [forwardInvocation M p row chunk isChunked]
  encode := encode
  observe := fun _ _ => observe

def backwardRead {α : Type} (p : Parameters) (row chunk : Nat) (wholeRow : Bool) :
    (i : Fin (backwardInputs p wholeRow).length) →
      ChannelValues α (backwardInputs p wholeRow) → TileIndex ((backwardInputs p wholeRow)[i]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0))))
  · intro _ _; exact row
  · intro _ j; exact row*p.logitsStride+column p wholeRow chunk j.1.val
  · intro _ _; exact row
  · intro _ _; exact row*p.upstreamStride

def backwardInputMask {α : Type} (p : Parameters) (chunk : Nat) (wholeRow : Bool) :
    (i : Fin (backwardInputs p wholeRow).length) →
      ChannelValues α (backwardInputs p wholeRow) → TileIndex ((backwardInputs p wholeRow)[i]).shape → Prop := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0))))
  · intro _ _; exact True
  · intro _ j; exact column p wholeRow chunk j.1.val < p.vocab
  · intro _ _; exact True
  · intro xs _; exact backwardLabel p wholeRow xs ≠ -100

def backwardInvocation {β : Type} (M : Algebra β) (p : Parameters)
    (row chunk : Nat) : Invocation β where
  kernel := KernelSource.backward p
  backend := engine M
  pids := fun ax => if ax=0 then row else chunk
  setPids := fun ax => decide (ax<2)

def backwardIO {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α)
    (p : Parameters) (row chunk : Nat) (wholeRow : Bool) : ProgramIO α β where
  inputs := backwardInputs p wholeRow
  outputs := backwardOutputs p
  scratch := []
  buffers := [p.labels.cast,p.logits,p.lse,p.upstream]
  read := backwardRead p row chunk wholeRow
  inputMask := backwardInputMask p chunk wholeRow
  write := by
    refine Fin.cases ?_ (fun k => k.elim0)
    intro _ j
    exact row*p.logitsStride+(chunk*p.block+j.1.val)
  outputMask := by
    refine Fin.cases ?_ (fun k => k.elim0)
    intro _ j
    exact chunk*p.block+j.1.val<p.vocab
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [backwardInvocation M p row chunk]
  encode := encode
  observe := fun _ _ => observe

end
end VeriTile.Bench.Repairs.FastCeLoss.PublicIO

namespace VeriTile.Bench.Repairs.FastCeLoss.LogicalIO
open VeriTile Triton FP.Structural
noncomputable section

open KernelSource
structure ForwardInputs (p : Parameters) where
  label : Int
  logits : Fin p.vocab → ℝ
  gatheredTarget : ℝ
structure ForwardOutputs where
  lse : ℝ
  loss : ℝ
structure BackwardInputs (p : Parameters) where
  label : Int
  logits : Fin p.vocab → ℝ
  savedLSE : ℝ
  upstream : ℝ

def forwardInputs (p : Parameters) : ChannelValues ℝ (PublicIO.forwardInputs p Bool.true) ≃ ForwardInputs p where
  toFun xs := ⟨xs (0 : Fin 3) (),fun j => xs (1 : Fin 3) (j,()),xs (2 : Fin 3) ()⟩
  invFun xs := Fin.cases (fun _ => xs.label) (Fin.cases (fun j => xs.logits j.1)
    (Fin.cases (fun _ => xs.gatheredTarget) (fun i => i.elim0)))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def forwardOutputs (p : Parameters) : ChannelValues ℝ (PublicIO.forwardOutputs p) ≃ ForwardOutputs where
  toFun ys := ⟨ys (0 : Fin 2) (),ys (1 : Fin 2) ()⟩
  invFun ys := Fin.cases (fun _ => ys.lse) (Fin.cases (fun _ => ys.loss) (fun i => i.elim0))
  left_inv := by intro ys; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro ys; cases ys; rfl

def backwardInputs (p : Parameters) : ChannelValues ℝ (PublicIO.backwardInputs p Bool.true) ≃ BackwardInputs p where
  toFun xs := ⟨xs (0 : Fin 4) (),fun j => xs (1 : Fin 4) (j,()),xs (2 : Fin 4) (),xs (3 : Fin 4) ()⟩
  invFun xs := Fin.cases (fun _ => xs.label) (Fin.cases (fun j => xs.logits j.1)
    (Fin.cases (fun _ => xs.savedLSE) (Fin.cases (fun _ => xs.upstream) (fun i => i.elim0))))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def backwardOutputs (p : Parameters) : ChannelValues ℝ (PublicIO.backwardOutputs p) ≃ (Fin p.block → ℝ) where
  toFun ys j := ys (0 : Fin 1) (j,())
  invFun ys := Fin.cases (fun j => ys j.1) (fun i => i.elim0)
  left_inv := by intro ys; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl

def forwardView (M : Algebra (WithBot ℝ)) (observe : WithBot ℝ → ℝ)
    (p : Parameters) (row chunk : Nat) (chunked : Bool) :
    LogicalProgramIO ℝ (WithBot ℝ) (ForwardInputs p) ForwardOutputs where
  core := PublicIO.forwardIO M some observe p row chunk chunked Bool.true
  inputs := forwardInputs p
  outputs := forwardOutputs p

def backwardView (M : Algebra (WithBot ℝ)) (observe : WithBot ℝ → ℝ)
    (p : Parameters) (row chunk : Nat) :
    LogicalProgramIO ℝ (WithBot ℝ) (BackwardInputs p) (Fin p.block → ℝ) where
  core := PublicIO.backwardIO M some observe p row chunk Bool.true
  inputs := backwardInputs p
  outputs := backwardOutputs p

/-- Total access to the logical vocabulary, with the original zero extension. -/
def logit {n : Nat} (xs : Fin n → ℝ) (j : Nat) : ℝ := if h : j<n then xs ⟨j,h⟩ else 0

end
end VeriTile.Bench.Repairs.FastCeLoss.LogicalIO
