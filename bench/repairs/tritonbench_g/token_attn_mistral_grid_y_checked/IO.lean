import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Ports
import bench.repairs.tritonbench_g.token_attn_mistral.Real
import bench.repairs.tritonbench_g.token_attn_mistral.Native
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.IO
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

def launch {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (pid : Grid.PID) : Invocation α where
  kernel := kernel p
  backend := withSum M R
  clearRegisters := Bool.true
  pids := fun ax=>if ax=0 then pid.1 else if ax=1 then pid.2 else 0
  numPids := some (fun _=>0)

def program {α : Type} (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) : ProgramIO α α where
  toMemoryIO := Ports.memory p work counts
  schedule := fun _=>work.map (launch M R p)
  encode := id
  observe := fun _ _=>id

/-- Actual affine-address candidate, on unchanged full public IO and grid. -/
def affineLaunch {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (pid : Grid.PID) : Invocation α :=
  { launch M R p pid with kernel := affineKernel p }

def affine {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat) :
    ProgramIO.Variant (program M R p work counts) where
  schedule := fun _ => work.map (affineLaunch M R p)
  observe := fun _ _ => id

def real (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :=
  program Real.algebra Real.sum p work counts

def native (R : Native.Reductions) (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :=
  program Native.algebra (Native.reduction R) p work counts

def observe (I : Native.Instructions) (S : Native.Schedule)
    (p : Parameters) (work : List Grid.PID) :
    (o : Fin (Ports.outputs p work).length) → TileIndex (Ports.outputs p work)[o].shape → Native.Term → Native.Word :=
  fun _ _=>Native.eval I S

end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.IO

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Logical
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

def inputs (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :
    (IO.real p work counts).Inputs ≃ Ports.Logical ℝ work.length (Ports.span p work counts) p.width where
  toFun := Ports.logical p work counts
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.sequence j.1
    | ⟨1,_⟩ => fun j => xs.start j.1
    | ⟨2,_⟩ => fun j => xs.request j.1
    | ⟨3,_⟩ => fun j => xs.attention j.1
    | ⟨4,_⟩ => fun j => xs.token j.1 j.2.1
    | ⟨5,_⟩ => fun j => xs.probability j.1 j.2.1
    | ⟨6,_⟩ => fun j => xs.value j.1 j.2.1 j.2.2.1
    | ⟨n+7,h⟩ => False.elim (by simp [IO.real, IO.program, Ports.memory, Ports.inputs] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j
    all_goals first | (rcases j with ⟨a,b,c,⟨⟩⟩; rfl) | (rcases j with ⟨a,b,⟨⟩⟩; rfl) | (rcases j with ⟨a,⟨⟩⟩; rfl)
  right_inv := by intro xs; cases xs; rfl

def outputs (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :
    (IO.real p work counts).Outputs ≃ (Fin work.length → Fin p.width → ℝ) where
  toFun := fun values job d => values
    ⟨0, by simp [IO.real,IO.program,Ports.memory,Ports.outputs]⟩ (job,d,())
  invFun := fun f port => match port with
    | ⟨0,_⟩ => fun j => f j.1 j.2.1
    | ⟨n+1,h⟩ => False.elim (by simp [IO.real,IO.program,Ports.memory,Ports.outputs] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port
    funext j
    rcases j with ⟨a,b,⟨⟩⟩
    rfl
  right_inv := by intro f; rfl

/-- The exact original metadata equality, on named logical fields. -/
def domain (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (xs : Ports.Logical ℝ work.length (Ports.span p work counts) p.width) : Prop :=
  ∀ job, xs.attention job = counts job

def io (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :
    LogicalProgramIO ℝ ℝ (Ports.Logical ℝ work.length (Ports.span p work counts) p.width)
      (Fin work.length → Fin p.width → ℝ) where
  core := IO.real p work counts
  inputs := inputs p work counts
  outputs := outputs p work counts

end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Logical
