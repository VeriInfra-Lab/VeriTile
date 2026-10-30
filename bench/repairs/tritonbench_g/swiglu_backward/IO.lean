import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.swiglu_backward.Proofs.WholePublic
import bench.tritonbench_g.swiglu_backward.Proofs.Observed
import bench.repairs.tritonbench_g.swiglu_backward.Kernels
import bench.tritonbench_g.swiglu_backward.Proofs.WholeGrid
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.SwigluBackward.IO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section

abbrev inputs (c : Layout) : List IOChannel :=
  [⟨.real,[c.rows,c.cols],c.X⟩,⟨.real,[c.rows,c.cols],c.Y⟩,⟨.real,[c.rows,c.cols],c.DOUT⟩]
abbrev outputs (c : Layout) : List IOChannel :=
  [⟨.real,[c.rows,c.cols],c.DX⟩,⟨.real,[c.rows,c.cols],c.DY⟩,⟨.real,[c.rows,c.cols],c.OUT⟩]

def read (c : Layout) (i : Fin (inputs c).length) :
    ChannelValues α (inputs c) → TileIndex (inputs c)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun _ j => j.1.val*c.sx+j.2.1.val
  | ⟨1,_⟩ => fun _ j => j.1.val*c.sy+j.2.1.val
  | ⟨2,_⟩ => fun _ j => j.1.val*c.sd+j.2.1.val
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

def write (c : Layout) (o : Fin (outputs c).length) :
    ChannelValues α (inputs c) → TileIndex (outputs c)[o].shape → Nat :=
  match o with
  | ⟨0,_⟩ => fun _ j => j.1.val*c.sdx+j.2.1.val
  | ⟨1,_⟩ => fun _ j => j.1.val*c.sdy+j.2.1.val
  | ⟨2,_⟩ => fun _ j => j.1.val*c.so+j.2.1.val
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

def outputMask (c : Layout) (R : Bool) (o : Fin (outputs c).length) :
    ChannelValues α (inputs c) → TileIndex (outputs c)[o].shape → Prop :=
  match o with
  | ⟨0,_⟩ | ⟨1,_⟩ => fun _ _ => True
  | ⟨2,_⟩ => fun _ _ => R = Bool.true
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

/-- The same selected FP32 source and program axes at every grid position. -/
def invocation (M : Algebra β) (c : Layout) (B : Nat) (R : Bool)
    (tile : Nat × Nat) : Invocation β where
  kernel := kernel c.X c.Y c.DOUT c.OUT c.DX c.DY c.sx c.sy c.sd c.so c.sdx c.sdy c.cols B R
  backend := M
  pids := fun ax => if ax=0 then tile.1 else if ax=1 then tile.2 else 0

/-- The existing integer-address candidate with the same complete source body. -/
def commutedInvocation (M : Algebra β) (c : Layout) (B : Nat) (R : Bool)
    (tile : Nat × Nat) : Invocation β where
  kernel := commutedKernel c.X c.Y c.DOUT c.OUT c.DX c.DY c.sx c.sy c.sd c.so c.sdx c.sdy c.cols B R
  backend := M
  pids := fun ax => if ax=0 then tile.1 else if ax=1 then tile.2 else 0

/-- Complete matrix snapshots and output channels for an explicitly ordered
full grid. The optional forward channel is inactive when recomputation is off.
Structural `.real` cells retain the source AST's FP32 casts and arithmetic. -/
abbrev swigluIO (M : Algebra β) (encode : α → β) (observe : β → α)
    (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) : ProgramIO α β where
  inputs := inputs c
  outputs := outputs c
  scratch := []
  buffers := [c.X,c.Y,c.DOUT,c.DX,c.DY,c.OUT]
  read := read c
  inputMask := fun _ _ _ => True
  write := write c
  outputMask := outputMask c R
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => tiles.map (invocation M c B R)
  encode := encode
  observe := fun _ _ => observe

/-- Same typed inputs, output observers and full-grid order; only the kernel's
integer pointer-address multiplication operands are commuted. -/
def commuted {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α)
    (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :
    ProgramIO.Variant (swigluIO M encode observe c B R tiles) where
  schedule := fun _ => tiles.map (commutedInvocation M c B R)
  observe := fun _ _ => observe

/-- Pointwise independent three-channel formula on logical matrix inputs. -/
def reference (c : Layout) (f : Channel → α → α → α → α)
    (xs : ChannelValues α (inputs c)) : ChannelValues α (outputs c) := fun o =>
  match o with
  | ⟨0,_⟩ => fun j => f .dx (xs 0 j) (xs 1 j) (xs 2 j)
  | ⟨1,_⟩ => fun j => f .dy (xs 0 j) (xs 1 j) (xs 2 j)
  | ⟨2,_⟩ => fun j => f .out (xs 0 j) (xs 1 j) (xs 2 j)
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

end
end VeriTile.Bench.Repairs.SwigluBackward.IO

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

abbrev swigluNativeIO (M : Observed.NativeModel (BitVec 32))
    (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :=
  IO.swigluIO Observed.graphAlgebra Observed.Term.input (Observed.interpret M) c B R tiles

end
end VeriTile.Bench.Repairs.SwigluBackward

namespace VeriTile.Bench.Repairs.SwigluBackward.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section

/-- Named logical snapshots; every original input channel is retained. -/
structure Inputs (c : Layout) where
  x : Fin (c.rows) → Fin (c.cols) → ℝ
  y : Fin (c.rows) → Fin (c.cols) → ℝ
  dout : Fin (c.rows) → Fin (c.cols) → ℝ

/-- Exact input coordinate change, including inactive lanes. -/
def inputs (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) : (IO.swigluIO (α := ℝ) realAlgebra id id c B R tiles).Inputs ≃ Inputs c where
  toFun := fun values => ⟨fun r c => values 0 (r,c,()), fun r c => values 1 (r,c,()), fun r c => values 2 (r,c,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.x j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.y j.1 j.2.1
    | ⟨2,_⟩ => fun j => xs.dout j.1 j.2.1
    | ⟨n+3,h⟩ => False.elim (by simp [IO.swigluIO, IO.inputs, IO.outputs] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Every declared output cell is retained, even when its source mask is inactive. -/
def outputs (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) : (IO.swigluIO (α := ℝ) realAlgebra id id c B R tiles).Outputs ≃ (Channel → Fin (c.rows) → Fin (c.cols) → ℝ) where
  toFun := fun values ch r c => match ch with
    | .dx => values 0 (r,c,())
    | .dy => values 1 (r,c,())
    | .out => values 2 (r,c,())
  invFun := fun f port => match port with
    | ⟨0,_⟩ => fun j => f .dx j.1 j.2.1
    | ⟨1,_⟩ => fun j => f .dy j.1 j.2.1
    | ⟨2,_⟩ => fun j => f .out j.1 j.2.1
    | ⟨n+3,h⟩ => False.elim (by simp [IO.swigluIO, IO.inputs, IO.outputs] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by
    intro f
    funext ch r c
    cases ch <;> rfl

def io (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :
    LogicalProgramIO ℝ ℝ (Inputs c) (Channel → Fin c.rows → Fin c.cols → ℝ) where
  core := IO.swigluIO realAlgebra id id c B R tiles
  inputs := inputs c B R tiles
  outputs := outputs c B R tiles

end
end VeriTile.Bench.Repairs.SwigluBackward.Logical
