import bench.tritonbench_g.nested_loops_processing.Kernels
import bench.tritonbench_g.nested_loops_processing.Proofs.Math
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.IO
open VeriTile.Triton FP.Structural
noncomputable section

abbrev inputs (count : Nat) : List IOChannel :=
  [⟨.int,[count],"input"⟩,⟨.int,[count],"output"⟩]
abbrev outputs (count : Nat) : List IOChannel := [⟨.int,[count],"output"⟩]

/-- A single complete original or commuted source. Program IDs are unused and
left unchanged; all nested loops, integer loads/stores and barriers remain. -/
def invocation (M : Algebra α) (candidate : Bool) (rows cols : Nat) : Invocation α where
  kernel := Kernels.kernel candidate "input" "output" (rows*cols) cols
  backend := M
  pids := fun _ => 0
  setPids := fun _ => Bool.false

/-- Input/output snapshots include every int32 allocation cell: unwritten
output cells therefore retain their declared zero initialization. -/
abbrev nestedIO (M : Algebra α) (candidate : Bool) (rows cols : Nat) : ProgramIO α α where
  inputs := inputs (rows*cols)
  outputs := outputs (rows*cols)
  scratch := []
  buffers := ["input","output"]
  read := fun i => Fin.cases (fun _ j => j.1.val)
    (fun k => Fin.cases (fun _ j => j.1.val) (fun z => z.elim0) k) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => j.1.val) (fun z => z.elim0) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation M candidate rows cols]
  encode := id
  observe := fun _ _ => id

/-- Original constructor data, including signed-int32 wrapping of arange. -/
def initialized (rows cols : Nat) (vals : ChannelValues α (inputs (rows*cols))) : Prop :=
  (∀ j, vals (0 : Fin 2) j = signedValue (arangeWord j.1.val)) ∧ (∀ j, vals (1 : Fin 2) j = (0 : Int))

/-- Lift an independent integer formula to the complete output channel. -/
def result (rows cols : Nat) (f : Nat → Int)
    (_ : ChannelValues α (inputs (rows*cols))) : ChannelValues α (outputs (rows*cols)) :=
  fun o => Fin.cases (fun j => f j.1.val) (fun z => z.elim0) o

def commuted (M : Algebra α) (rows cols : Nat) :
    ProgramIO.Variant (nestedIO M Bool.false rows cols) where
  schedule := fun _ => [invocation M Bool.true rows cols]
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.IO

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.PairIO
open VeriTile Triton FP.Structural
noncomputable section

/-- The complete single-launch source with the original exact int32 input
cells and zero-initialized output snapshot. -/
def original (M : Algebra α) (rows cols : Nat) :=
  ProgramIO.Implementation.original (IO.nestedIO M Bool.false rows cols)
    (IO.initialized rows cols) .exactCells

/-- The existing complete candidate changes its initial integer index product.
All public cells, initialization conditions, source loops and barriers persist. -/
def indexCommuted (M : Algebra α) (rows cols : Nat) :=
  ProgramIO.Implementation.ofVariant (IO.commuted M rows cols)
    (IO.initialized rows cols) .exactCells

def assumptions (M : Algebra α) (rows cols : Nat) :
    Spec.Assumptions (ProgramIO.Implementation (IO.nestedIO M Bool.false rows cols)
      (IO.initialized rows cols) .exactCells) := []

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.PairIO

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.LogicalView
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Matrix (rows cols : Nat) := Fin rows → Fin cols → Int
structure Inputs (rows cols : Nat) where
  input : Matrix rows cols
  initialOutput : Matrix rows cols

def matrixEquiv (rows cols : Nat) : (TileIndex [rows*cols] → Int) ≃ Matrix rows cols where
  toFun := fun xs r c => xs (finProdFinEquiv (r,c),())
  invFun := fun xs j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs; funext j
    change xs (finProdFinEquiv (finProdFinEquiv.symm j.1),())=xs j
    rw [Equiv.apply_symm_apply]
  right_inv := by intro xs; funext r c; change xs (finProdFinEquiv.symm (finProdFinEquiv (r,c))).1 (finProdFinEquiv.symm (finProdFinEquiv (r,c))).2=xs r c; rw [Equiv.symm_apply_apply]

def inputEquiv (rows cols : Nat) : ChannelValues α (IO.inputs (rows*cols)) ≃ Inputs rows cols where
  toFun := fun xs => ⟨matrixEquiv rows cols (xs 0),matrixEquiv rows cols (xs 1)⟩
  invFun := fun xs => Fin.cases ((matrixEquiv rows cols).symm xs.input)
    (Fin.cases ((matrixEquiv rows cols).symm xs.initialOutput) (fun k => k.elim0))
  left_inv := by intro xs; funext o; fin_cases o <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro xs
    cases xs with
    | mk x y =>
      change Inputs.mk ((matrixEquiv rows cols) ((matrixEquiv rows cols).symm x))
        ((matrixEquiv rows cols) ((matrixEquiv rows cols).symm y)) = _
      simp only [Equiv.apply_symm_apply]

def outputEquiv (rows cols : Nat) : ChannelValues α (IO.outputs (rows*cols)) ≃ Matrix rows cols :=
  (ChannelValues.singleEquiv α ⟨.int,[rows*cols],"output"⟩).trans (matrixEquiv rows cols)

/-- The full old exact-cell judgment remains fixed, including both snapshots. -/
def nestedIO (M : Algebra α) (rows cols : Nat) [Inhabited α] :
    LogicalDomainIO (IO.nestedIO M Bool.false rows cols).Inputs (IO.nestedIO M Bool.false rows cols).Outputs
      (Inputs rows cols) (Matrix rows cols) (IO.nestedIO M Bool.false rows cols).ImplementsOnCells where
  inputs := inputEquiv rows cols
  outputs := outputEquiv rows cols
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.LogicalView
