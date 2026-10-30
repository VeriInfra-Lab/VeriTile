import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.isfinite_kernel.Proofs.Source
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Shared public strided classification IO. Both reader entries use the same
layout and source identity; correctness and floating equivalence stay separate.
The source AST retains FP32/FP64 input storage. Its Structural floating payload
uses the real-tagged carrier, and its output is a Boolean cell.
-/
namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Triton FP.Structural
noncomputable section

/-- Exact masked input/output windows, including signed byte strides and
nonzero pointer offsets. The selected scalar classifier is a primitive-model
parameter; no finiteness precondition is attached to its inputs. -/
def classificationIO {A : Type} [Inhabited A] (precision : Precision)
    (classify : A → Bool) (input output : View) (n B pid : Nat) : ProgramIO A A where
  inputs := [⟨.real,[B],"x"⟩]
  outputs := [⟨.bool,[B],"y"⟩]
  scratch := []
  buffers := ["x","y"]
  read := by intro i; have h : i=(0 : Fin 1) := Fin.eq_zero i; subst i; exact fun _ j => input.address (pid*B+j.1.val)
  inputMask := by intro i; have h : i=(0 : Fin 1) := Fin.eq_zero i; subst i; exact fun _ j => pid*B+j.1.val<n
  write := by intro i; have h : i=(0 : Fin 1) := Fin.eq_zero i; subst i; exact fun _ j => output.address (pid*B+j.1.val)
  outputMask := by intro i; have h : i=(0 : Fin 1) := Fin.eq_zero i; subst i; exact fun _ j => pid*B+j.1.val<n
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{
    kernel := source precision input output n B
    backend := Source.algebra precision classify
    pids := fun ax => if ax=0 then pid else 0 }]
  encode := id
  observe := fun _ _ => id

/-- The genuine existing integer-index candidate has the same public layout
and primitive interpretation. Only its selected source schedule differs. -/
def indexCandidate {A : Type} [Inhabited A] (precision : Precision)
    (classify : A → Bool) (input output : View) (n B pid : Nat) :
    ProgramIO.Variant (classificationIO precision classify input output n B pid) where
  schedule := fun _ => [{
    kernel := indexCommuted precision input output n B
    backend := Source.algebra precision classify
    pids := fun ax => if ax=0 then pid else 0 }]
  observe := fun _ _ => id

/-- Actual original invocation and its public layout on the unrestricted
logical input domain. The program view stores the schedule and observer. -/
def originalProgram {A : Type} [Inhabited A] (precision : Precision)
    (classify : A → Bool) (input output : View) (n B pid : Nat) :=
  ProgramIO.Implementation.original
    (classificationIO precision classify input output n B pid) (fun _ => True)

/-- The existing index-commuted invocation with the identical public IO. -/
def commutedProgram {A : Type} [Inhabited A] (precision : Precision)
    (classify : A → Bool) (input output : View) (n B pid : Nat) :=
  ProgramIO.Implementation.ofVariant
    (indexCandidate precision classify input output n B pid) (fun _ => True)

/-- Independent pointwise mathematical classifier on the logical input tile. -/
def classification {A : Type} [Inhabited A] (precision : Precision)
    (classify : A → Bool) (input output : View) (n B pid : Nat)
    (xs : (classificationIO precision classify input output n B pid).Inputs) :
    (classificationIO precision classify input output n B pid).Outputs :=
  by
    intro o
    have h : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    exact fun j => classify (xs (0 : Fin 1) j)

end
end VeriTile.Bench.Repairs.IsfiniteKernel

namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Triton FP.Structural
noncomputable section

def logicalInputs {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat) :
    (classificationIO precision classify input output n B pid).Inputs ≃ (Fin B → A) where
  toFun xs := fun i => xs ⟨0,by simp [classificationIO]⟩ (i,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun i => xs i.1
  left_inv := by intro xs; funext o; fin_cases o; funext i; rcases i with ⟨i,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def logicalOutputs {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat) :
    (classificationIO precision classify input output n B pid).Outputs ≃ (Fin B → Bool) where
  toFun xs := fun i => xs ⟨0,by simp [classificationIO]⟩ (i,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun i => xs i.1
  left_inv := by intro xs; funext o; fin_cases o; funext i; rcases i with ⟨i,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl
end
end VeriTile.Bench.Repairs.IsfiniteKernel

namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Triton FP.Structural
noncomputable section
def logicalIO {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat) : LogicalProgramIO A A (Fin B → A) (Fin B → Bool) where
  core := classificationIO precision classify input output n B pid
  inputs := logicalInputs precision classify input output n B pid
  outputs := logicalOutputs precision classify input output n B pid
end
end VeriTile.Bench.Repairs.IsfiniteKernel
