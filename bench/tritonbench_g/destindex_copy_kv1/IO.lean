import bench.tritonbench_g.destindex_copy_kv1.Proofs.Tile
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Shared public finite-row IO data for destination-indexed copies. The source
and metadata have separate typed channels; writes use the admitted logical
signed destination values. No program result is stored in this descriptor. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- FP32 payloads use the source's erased `.real` channel; destination indices
remain signed integers. The actual source fixes storage and cast precision. -/
def inputChannels (c : Config) : List IOChannel :=
  [⟨.real,[c.count,c.heads,c.dims],inputRegion⟩,⟨.int,[c.count],metadataRegion.cast⟩]

def outputChannels (c : Config) : List IOChannel :=
  [⟨.real,[c.count,c.heads,c.dims],outputRegion⟩]

abbrev Inputs (α : Type) (c : Config) := ChannelValues α (inputChannels c)
abbrev Outputs (α : Type) (c : Config) := ChannelValues α (outputChannels c)

def destinationWord (c : Config) (values : Inputs α c) (row : Fin c.count) : Int :=
  values ⟨1,by simp [inputChannels]⟩ (row,PUnit.unit)

/-- Signed storage represents the same nonnegative injective destination map
as the earlier Nat-index theorem. This is a logical metadata condition. -/
def Admissible (c : Config) (values : Inputs α c) : Prop :=
  (∀ row,0≤destinationWord c values row) ∧ Function.Injective (destinationWord c values)

def destination (c : Config) (values : Inputs α c) (row : Nat) : Nat :=
  if h : row<c.count then (destinationWord c values ⟨row,h⟩).toNat else 0

def source {α : Type} [Inhabited α] (c : Config) (values : Inputs α c)
    (row head dim : Nat) : α :=
  if hr : row<c.count then if hh : head<c.heads then if hd : dim<c.dims then
    values ⟨0,by simp [inputChannels]⟩ (⟨row,hr⟩,⟨head,hh⟩,⟨dim,hd⟩,PUnit.unit)
  else default else default else default

/-- Independent row-copy formula on logical arrays. Output row `r` is wired
by the descriptor to destination[r]; every head/dimension preserves its value. -/
def target (c : Config) (values : Inputs α c) : Outputs α c :=
  Fin.cases (values ⟨0,by simp [inputChannels]⟩) (fun o => Fin.elim0 o)

/-- Finite windows include the unmasked scalar destination loads, even when
one payload dimension is empty. Only the declared output scatter may change. -/
def memory (c : Config) : MemoryIO α where
  inputs := inputChannels c
  outputs := outputChannels c
  buffers := [inputRegion,metadataRegion.cast,outputRegion]
  read := Fin.cases (fun _ j => flat c j.1.val j.2.1.val j.2.2.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun values j =>
    flat c (destinationWord c values j.1).toNat j.2.1.val j.2.2.1.val)
    (fun o => Fin.elim0 o)
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- One actual selected FP32 source invocation. Other axes are zero and unused. -/
def invocation (rest : Algebra α) (c : Config) (pid : Nat) : Invocation α where
  kernel := kernel (layout c)
  backend := Tile.rawBackend rest
  pids := fun ax => if ax=0 then pid else 0

/-- Complete arbitrary-order copy grid, preserving payloads without a floating
rewrite or a finiteness restriction. Bounds are supplied to the shared judgment. -/
def copyIO (rest : Algebra α) (c : Config) (ps : List Nat) : ProgramIO α α where
  toMemoryIO := memory c
  schedule := fun _ => ps.map (invocation rest c)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Logical payload and signed scatter metadata retain every declared input. -/
structure LogicalInputs (α : Type) (c : Config) where
  payload : Fin c.count → Fin c.heads → Fin c.dims → α
  destinations : Fin c.count → Int

abbrev LogicalOutputs (α : Type) (c : Config) :=
  Fin c.count → Fin c.heads → Fin c.dims → α

def inputEquiv (α : Type) (c : Config) : Inputs α c ≃ LogicalInputs α c where
  toFun := fun values => ⟨fun row head dim => values ⟨0,by simp [inputChannels]⟩ (row,head,dim,()),
    fun row => values ⟨1,by simp [inputChannels]⟩ (row,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.payload j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => xs.destinations j.1) (fun o => Fin.elim0 o))
  left_inv := by
    intro values
    funext o j
    fin_cases o
    · rcases j with ⟨row,head,dim,u⟩
      cases u
      rfl
    · rcases j with ⟨row,u⟩
      cases u
      rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (α : Type) (c : Config) : Outputs α c ≃ LogicalOutputs α c where
  toFun := fun values row head dim => values ⟨0,by simp [outputChannels]⟩ (row,head,dim,())
  invFun := fun xs => Fin.cases (fun j => xs j.1 j.2.1 j.2.2.1) (fun o => Fin.elim0 o)
  left_inv := by
    intro values
    funext o j
    fin_cases o
    rcases j with ⟨row,head,dim,u⟩
    cases u
    rfl
  right_inv := by intro xs; rfl

def LogicalAdmissible (c : Config) (xs : LogicalInputs α c) : Prop :=
  (∀ row, 0 ≤ xs.destinations row) ∧ Function.Injective xs.destinations

/-- Exact existing scatter schedule and memory judgment in logical coordinates. -/
def logicalIO (rest : Algebra α) (c : Config) (ps : List Nat) :
    LogicalProgramIO α α (LogicalInputs α c) (LogicalOutputs α c) where
  core := copyIO rest c ps
  inputs := inputEquiv α c
  outputs := outputEquiv α c

end
end VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
