import bench.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.LayerNormWelfold.IO
open VeriTile.Triton FP.Structural Reuse
noncomputable section

abbrev inputs (S D : Nat) : List IOChannel :=
  [⟨.real,[S,D],"X"⟩,⟨.real,[D],"W"⟩,⟨.real,[D],"Bias"⟩]
abbrev outputs (Y Mean Rstd : RegionName) (S D : Nat) : List IOChannel :=
  [⟨.real,[S],Mean⟩,⟨.real,[S],Rstd⟩,⟨.bf16,[S,D],Y⟩]

def invocation (M : Algebra α) (Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) (row : Nat) : Invocation α where
  kernel := SourceCode.rowKernel "X" "W" "Bias" Y Mean Rstd S D B (SourceBinding.tree tree)
  backend := withEquality M (fun _ _ => Bool.false)
  pids := fun ax => if ax=0 then row else 0

/-- Complete row grid with FP32 input/statistics and BF16 Y output. The
reduction tree fixes arithmetic topology; it supplies no expected values. -/
abbrev layerNormIO (M : Algebra α) (observe : α → α)
    (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B) : ProgramIO α α where
  inputs := inputs S D
  outputs := outputs Y Mean Rstd S D
  scratch := []
  buffers := ["X","W","Bias",Y,Mean,Rstd]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => j.2.1.val+D*j.1.val
    | ⟨1,_⟩ | ⟨2,_⟩ => fun j => j.1.val
    | ⟨_+3,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.1.val
    | ⟨2,_⟩ => fun j => j.2.1.val+D*j.1.val
    | ⟨_+3,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range S).map (invocation M Y Mean Rstd S D B tree)
  encode := id
  observe := fun _ _ => observe

def arrayX [Inhabited α] (S D : Nat) (vals : ChannelValues α (inputs S D)) (i : Nat) : α :=
  if row : i/D < S then if col : i%D < D then vals 0 (⟨i/D,row⟩,⟨i%D,col⟩,()) else default else default
def arrayW [Inhabited α] (S D : Nat) (vals : ChannelValues α (inputs S D)) (i : Nat) : α :=
  if col : i < D then vals 1 (⟨i,col⟩,()) else default
def arrayB [Inhabited α] (S D : Nat) (vals : ChannelValues α (inputs S D)) (i : Nat) : α :=
  if col : i < D then vals 2 (⟨i,col⟩,()) else default

/-- Logical output tuple: mean, inverse standard deviation, normalized Y. -/
def result [Inhabited α] (Y Mean Rstd : RegionName) (S D : Nat)
    (f : (Nat → α) → (Nat → α) → (Nat → α) → Nat → α × α × (Nat → α))
    (vals : ChannelValues α (inputs S D)) : ChannelValues α (outputs Y Mean Rstd S D) :=
  let ref := f (arrayX S D vals) (arrayW S D vals) (arrayB S D vals)
  fun o => match o with
    | ⟨0,_⟩ => fun j => (ref j.1.val).1
    | ⟨1,_⟩ => fun j => (ref j.1.val).2.1
    | ⟨2,_⟩ => fun j => (ref j.1.val).2.2 j.2.1.val
    | ⟨_+3,h⟩ => False.elim (by simp at h)

end
end VeriTile.Bench.Repairs.LayerNormWelfold.IO

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section

abbrev tree {B : Nat} : ReductionTree B → SourceCode.ReductionTree B := SourceBinding.tree

end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable

namespace VeriTile.Bench.Repairs.LayerNormWelfold.PairIO
open VeriTile.Triton FP.Structural Reuse
noncomputable section

def invocation (M : Algebra α) (reuse : Bool) (Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (row : Nat) : Invocation α where
  kernel := SourceCode.selected reuse "X" "W" "Bias" Y Mean Rstd S D B (Readable.tree tree)
  backend := withEquality M (fun _ _ => Bool.false)
  pids := fun ax => if ax=0 then row else 0

/-- The complete readable source runs on every row with the original three
input channels, three outputs, FP32 statistics and BF16 output cells. -/
abbrev layerNormIO (M : Algebra α) (observe : α → α) (reuse : Bool)
    (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B) : ProgramIO α α :=
  {IO.layerNormIO M observe Y Mean Rstd S D B tree with
    schedule := fun _ => (List.range S).map (invocation M reuse Y Mean Rstd S D B tree)}

def candidate (M : Algebra α) (Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ProgramIO.Variant (layerNormIO M id Bool.false Y Mean Rstd S D B tree) where
  schedule := fun _ => (List.range S).map (invocation M Bool.true Y Mean Rstd S D B tree)
  observe := fun _ _ => id

def original (M : Algebra α) (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B) :=
  ProgramIO.Implementation.original (layerNormIO M id Bool.false Y Mean Rstd S D B tree) (fun _ => True)
def rowOffsetReused (M : Algebra α) (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B) :=
  ProgramIO.Implementation.ofVariant (candidate M Y Mean Rstd S D B tree) (fun _ => True)
end
end VeriTile.Bench.Repairs.LayerNormWelfold.PairIO

namespace VeriTile.Bench.Repairs.LayerNormWelfold.LogicalIO
open VeriTile.Triton FP.Structural Reuse
noncomputable section

structure Inputs (S D : Nat) where
  x : Fin S → Fin D → ℝ
  weight : Fin D → ℝ
  bias : Fin D → ℝ

def inputs (A : VeriTile.Triton.FP.Structural.Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat) (B : Nat) (tree : ReductionTree B) : (PairIO.layerNormIO A id Bool.false Y Mean Rstd S D B tree).Inputs ≃ Inputs S D where
  toFun := fun v => ⟨fun i0 i1 => v (0 : Fin 3) (i0,i1,()), fun i0 => v (1 : Fin 3) (i0,()), fun i0 => v (2 : Fin 3) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.x j.1 j.2.1
    | ⟨1,_⟩ => fun j => v.weight j.1
    | ⟨2,_⟩ => fun j => v.bias j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (S D : Nat) where
  mean : Fin S → ℝ
  inverse : Fin S → ℝ
  y : Fin S → Fin D → ℝ

def outputs (A : VeriTile.Triton.FP.Structural.Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat) (B : Nat) (tree : ReductionTree B) : (PairIO.layerNormIO A id Bool.false Y Mean Rstd S D B tree).Outputs ≃ Outputs S D where
  toFun := fun v => ⟨fun i0 => v (0 : Fin 3) (i0,()), fun i0 => v (1 : Fin 3) (i0,()), fun i0 i1 => v (2 : Fin 3) (i0,i1,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.mean j.1
    | ⟨1,_⟩ => fun j => v.inverse j.1
    | ⟨2,_⟩ => fun j => v.y j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (A : VeriTile.Triton.FP.Structural.Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat) (B : Nat) (tree : ReductionTree B) : LogicalProgramIO ℝ ℝ (Inputs S D) (Outputs S D) where
  core := PairIO.layerNormIO A id Bool.false Y Mean Rstd S D B tree
  inputs := inputs A Y Mean Rstd S D B tree
  outputs := outputs A Y Mean Rstd S D B tree

end
end VeriTile.Bench.Repairs.LayerNormWelfold.LogicalIO
