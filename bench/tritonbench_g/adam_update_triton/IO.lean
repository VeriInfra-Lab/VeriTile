import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.swiglu_triton.Proofs.Math
import bench.tritonbench_g.adam_update_triton.Proofs.Tile
import VeriTile.Triton.Float.ProgramIO

/-! Shared IO data for the actual Lion source: three input windows and two
in-place output windows. The scalar ABI is a list of four typed registers.
The Real and native entries use these same source, mask and memory roles. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (B : Nat) : List IOChannel :=
  [⟨.real, [B], "P"⟩, ⟨.real, [B], "G"⟩, ⟨.real, [B], "M"⟩]

def outputs (B : Nat) : List IOChannel :=
  [⟨.real, [B], "P"⟩, ⟨.real, [B], "M"⟩]

abbrev Inputs (α : Type) (B : Nat) := ChannelValues α (inputs B)
abbrev Outputs (α : Type) (B : Nat) := ChannelValues α (outputs B)

def inputLane (B : Nat) : (i : Fin (inputs B).length) →
    TileIndex (inputs B)[i].shape → Fin B
  | ⟨0, _⟩ => fun j => j.1
  | ⟨1, _⟩ => fun j => j.1
  | ⟨2, _⟩ => fun j => j.1
  | ⟨i + 3, h⟩ => False.elim (by simp [inputs] at h)

def outputLane (B : Nat) : (o : Fin (outputs B).length) →
    TileIndex (outputs B)[o].shape → Fin B
  | ⟨0, _⟩ => fun j => j.1
  | ⟨1, _⟩ => fun j => j.1
  | ⟨i + 2, h⟩ => False.elim (by simp [outputs] at h)

/-- P and M occur as both inputs and outputs; G remains read-only. Only
active lanes are constrained. The source cell type is erased Real and its
default computation/storage precision is the explicit FP32 engine. -/
def memory (α : Type) (n B pid : Nat) : MemoryIO α where
  inputs := inputs B
  outputs := outputs B
  scratch := []
  buffers := ["P", "G", "M"]
  read := fun i _ j => pid * B + (inputLane B i j).val
  inputMask := fun i _ j => pid * B + (inputLane B i j).val < n
  write := fun o _ j => pid * B + (outputLane B o j).val
  outputMask := fun o _ j => pid * B + (outputLane B o j).val < n
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Runtime scalar arguments are bound before the actual source body. No
scalar is substituted into the source as a mathematical literal. -/
def invocation {α : Type} (sem : Tile.Semantics α) (params : Scalars α)
    (total n B pid : Nat) : Invocation α where
  kernel := source total n B
  backend := sem.backend
  arguments := [⟨"lr", .real, [], fun _ => params.lr⟩,
    ⟨"wd", .real, [], fun _ => params.wd⟩,
    ⟨"beta1", .real, [], fun _ => params.beta1⟩,
    ⟨"beta2", .real, [], fun _ => params.beta2⟩]
  pids := fun ax => if ax = 0 then pid else 0

def programIO {α β : Type} (sem : Tile.Semantics β) (params : Scalars β)
    (encode : α → β) (observe : β → α) (total n B pid : Nat) : ProgramIO α β :=
  { memory α n B pid with
    schedule := fun _ => [invocation sem params total n B pid]
    encode := encode
    observe := fun _ _ => observe }

def parameter {α : Type} {B : Nat} (values : Inputs α B) (j : Fin B) : α :=
  values ⟨0, by simp [inputs]⟩ (j, PUnit.unit)
def gradient {α : Type} {B : Nat} (values : Inputs α B) (j : Fin B) : α :=
  values ⟨1, by simp [inputs]⟩ (j, PUnit.unit)
def momentum {α : Type} {B : Nat} (values : Inputs α B) (j : Fin B) : α :=
  values ⟨2, by simp [inputs]⟩ (j, PUnit.unit)

/-- Pack the two independently stated result arrays in their declared order. -/
def result {α : Type} {B : Nat} (p m : Fin B → α) : Outputs α B
  | ⟨0, _⟩ => fun j => p j.1
  | ⟨1, _⟩ => fun j => m j.1
  | ⟨i + 2, h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Interface

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Real
open VeriTile.Triton FP.Structural
open Classical
noncomputable section

def semantics : Tile.Semantics ℝ where
  algebra := { VeriTile.Bench.Repairs.SwigluTriton.Math.algebra with
    unary := fun _ op x => match op with | .neg _ => -x | _ => 0 }
  less := fun a b => decide (a<b)
  equal := fun a b => decide (a=b)

end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Real

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (B : Nat) where
  parameter : Fin B → ℝ
  gradient : Fin B → ℝ
  momentum : Fin B → ℝ

structure Outputs (B : Nat) where
  parameter : Fin B → ℝ
  momentum : Fin B → ℝ

def inputsEquiv (B : Nat) : Interface.Inputs ℝ B ≃ Inputs B where
  toFun := fun values => ⟨Interface.parameter values,Interface.gradient values,Interface.momentum values⟩
  invFun := fun values => Fin.cases (fun j => values.parameter j.1)
    (Fin.cases (fun j => values.gradient j.1)
      (Fin.cases (fun j => values.momentum j.1) (fun i => Fin.elim0 i)))
  left_inv := by
    intro values
    funext i j
    fin_cases i <;> rcases j with ⟨j,⟨⟩⟩ <;> rfl
  right_inv := fun _ => rfl

def outputsEquiv (B : Nat) : Interface.Outputs ℝ B ≃ Outputs B where
  toFun := fun values => ⟨fun j => values (0 : Fin 2) (j,()), fun j => values (1 : Fin 2) (j,())⟩
  invFun := fun values => Interface.result values.parameter values.momentum
  left_inv := by
    intro values
    funext o j
    fin_cases o <;> rcases j with ⟨j,⟨⟩⟩ <;> rfl
  right_inv := fun _ => rfl

def io (params : Scalars ℝ) (total n B pid : Nat) :
    LogicalProgramIO ℝ ℝ (Inputs B) (Outputs B) where
  core := Interface.programIO Real.semantics params id id total n B pid
  inputs := inputsEquiv B
  outputs := outputsEquiv B

end
end VeriTile.Bench.Repairs.AdamUpdateTriton.LogicalIO
