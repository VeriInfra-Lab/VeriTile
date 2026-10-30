import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.swiglu_triton.Proofs.Math
import bench.tritonbench_g.swiglu_triton.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.SwigluTriton.Pair
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

/-- Full FP32 contiguous-row sources. The unchanged program ID selects the
actual row; forward writes C and backward overwrites A/B in source order. -/
def original (backward : Bool) (block cols : Nat) : Program where
  kernel := directSource backward block cols
  domain := []

def indexCommuted (backward : Bool) (block cols : Nat) : Program :=
  { original backward block cols with kernel := directIndexCommutedSource backward block cols }

end
end VeriTile.Bench.Repairs.SwigluTriton.Pair

namespace VeriTile.Bench.Repairs.SwigluTriton.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

abbrev inputChannels (block : Nat) : List IOChannel :=
  [⟨.real, [block], "A"⟩, ⟨.real, [block], "B"⟩, ⟨.real, [block], "DC"⟩]
abbrev outputChannels (block : Nat) : List IOChannel :=
  [⟨.real, [block], "C"⟩, ⟨.real, [block], "A"⟩, ⟨.real, [block], "B"⟩]

def inputRead (block cols row : Nat) (i : Fin (inputChannels block).length) :
    ChannelValues α (inputChannels block) → TileIndex (inputChannels block)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ => fun _ j => row*cols+j.1.val
  | ⟨_+3, h⟩ => False.elim (by simpa using h)

def inputMask (backward : Bool) (block cols : Nat) (i : Fin (inputChannels block).length) :
    ChannelValues α (inputChannels block) → TileIndex (inputChannels block)[i].shape → Prop :=
  match i with
  | ⟨0, _⟩ | ⟨1, _⟩ => fun _ j => j.1.val < cols
  | ⟨2, _⟩ => fun _ j => backward = Bool.true ∧ j.1.val < cols
  | ⟨_+3, h⟩ => False.elim (by simpa using h)

def outputWrite (block cols row : Nat) (i : Fin (outputChannels block).length) :
    ChannelValues α (inputChannels block) → TileIndex (outputChannels block)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ => fun _ j => row*cols+j.1.val
  | ⟨_+3, h⟩ => False.elim (by simpa using h)

def outputMask (backward : Bool) (block cols : Nat) (i : Fin (outputChannels block).length) :
    ChannelValues α (inputChannels block) → TileIndex (outputChannels block)[i].shape → Prop :=
  match i with
  | ⟨0, _⟩ => fun _ j => backward = Bool.false ∧ j.1.val < cols
  | ⟨1, _⟩ | ⟨2, _⟩ => fun _ j => backward = Bool.true ∧ j.1.val < cols
  | ⟨_+3, h⟩ => False.elim (by simpa using h)

/-- Exactly one selected source invocation; the row register convention is
the original program-ID convention, with no extra memory initialization. -/
def invocation (M : Algebra β) (backward : Bool) (block cols row : Nat) : Invocation β where
  kernel := directSource backward block cols
  backend := M
  pids := fun ax => if ax = 0 then row else 0

/-- Reusable typed IO assembly. The reader entries expose these same named
source, channel, window and mask fields with their selected interpretation. -/
abbrev io (M : Algebra β) (encode : α → β) (observe : β → α)
    (backward : Bool) (block cols row : Nat) : ProgramIO α β where
  inputs := inputChannels block
  outputs := outputChannels block
  scratch := []
  buffers := ["A", "B", "DC", "C"]
  read := inputRead block cols row
  inputMask := inputMask backward block cols
  write := outputWrite block cols row
  outputMask := outputMask backward block cols
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation M backward block cols row]
  encode := encode
  observe := fun _ _ => observe

def formula (f : Channel → α → α → α → α) (block : Nat)
    (xs : ChannelValues α (inputChannels block)) : ChannelValues α (outputChannels block) := fun o =>
  match o with
  | ⟨0, _⟩ => fun j => f .c (xs ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ j) (xs ⟨1, by simp [io, inputChannels, outputChannels, enabled]⟩ j) (xs ⟨2, by simp [io, inputChannels, outputChannels, enabled]⟩ j)
  | ⟨1, _⟩ => fun j => f .da (xs ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ j) (xs ⟨1, by simp [io, inputChannels, outputChannels, enabled]⟩ j) (xs ⟨2, by simp [io, inputChannels, outputChannels, enabled]⟩ j)
  | ⟨2, _⟩ => fun j => f .db (xs ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ j) (xs ⟨1, by simp [io, inputChannels, outputChannels, enabled]⟩ j) (xs ⟨2, by simp [io, inputChannels, outputChannels, enabled]⟩ j)
  | ⟨_+3, h⟩ => False.elim (by simpa using h)
end
end VeriTile.Bench.Repairs.SwigluTriton.PublicIO

namespace VeriTile.Bench.Repairs.SwigluTriton.Logical
open VeriTile.Triton FP.Structural

noncomputable section

/-- Named logical snapshots; every original input channel is retained. -/
structure Inputs (block : Nat) where
  a : Fin block → ℝ
  b : Fin block → ℝ
  dc : Fin block → ℝ

/-- Exact input coordinate change, including inactive lanes. -/
def inputs (backward : Bool) (block cols row : Nat) : (PublicIO.io (α := ℝ) Math.algebra id id backward block cols row).Inputs ≃ Inputs block where
  toFun := fun values => ⟨fun i => values 0 (i,()), fun i => values 1 (i,()), fun i => values 2 (i,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.a j.1
    | ⟨1,_⟩ => fun j => xs.b j.1
    | ⟨2,_⟩ => fun j => xs.dc j.1
    | ⟨n+3,h⟩ => False.elim (by simp [PublicIO.io, PublicIO.inputChannels, PublicIO.outputChannels] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j <;> rcases j with ⟨i,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Every declared output cell is retained, even when its source mask is inactive. -/
def outputs (backward : Bool) (block cols row : Nat) : (PublicIO.io (α := ℝ) Math.algebra id id backward block cols row).Outputs ≃ (Channel → Fin block → ℝ) where
  toFun := fun values ch i => match ch with
    | .c => values 0 (i,())
    | .da => values 1 (i,())
    | .db => values 2 (i,())
  invFun := fun f port => match port with
    | ⟨0,_⟩ => fun j => f .c j.1
    | ⟨1,_⟩ => fun j => f .da j.1
    | ⟨2,_⟩ => fun j => f .db j.1
    | ⟨n+3,h⟩ => False.elim (by simp [PublicIO.io, PublicIO.inputChannels, PublicIO.outputChannels] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j <;> rcases j with ⟨i,⟨⟩⟩ <;> rfl
  right_inv := by
    intro f
    funext ch i
    cases ch <;> rfl

def io (backward : Bool) (block cols row : Nat) :
    LogicalProgramIO ℝ ℝ (Inputs block) (Channel → Fin block → ℝ) where
  core := PublicIO.io Math.algebra id id backward block cols row
  inputs := inputs backward block cols row
  outputs := outputs backward block cols row

end
end VeriTile.Bench.Repairs.SwigluTriton.Logical
