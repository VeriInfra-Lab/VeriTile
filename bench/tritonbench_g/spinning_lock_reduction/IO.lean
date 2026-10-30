import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Real
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Tactic.FinCases
import bench.tritonbench_g.spinning_lock_reduction.Formula
import bench.tritonbench_g.spinning_lock_reduction.Proofs.ConcurrentKernel

/-! Public finite IO for the two original source stages. P/C/locks are named,
separate regions; each input aliases its corresponding returned output. P and
locks expose the actual source prefixes, and allocation padding stays framed.
Both Real and FP32 instantiate this same source/geometry/control description. -/
namespace VeriTile.Bench.Repairs.SpinningLock.IO
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

def partials (h : HostArgs) : Concurrent.Window :=
  {name := "partials",region := 0,count := Formula.partialCount h}

def output (count : Nat) : Concurrent.Window :=
  {name := "output",region := 1,count := count}

def locks (h : HostArgs) : Concurrent.Window :=
  {name := "locks",region := 0,count := h.numSms}

def io (V : Type) (h : HostArgs) (outputCount : Nat) (stage : Phased.Stage) :
    Concurrent.EventProgramIO (List SourceReader.Statement) V V V where
  source := kernel
  decoded := ConcurrentSource.decode h stage kernel
  scalarInputs := [⟨.mem,partials h⟩,⟨.mem,output outputCount⟩]
  wordInputs := [locks h]
  scalarOutputs := [partials h,output outputCount]
  wordOutputs := [locks h]
  encode := id
  observe := fun _ _ => id

def inputs (h : HostArgs) (outputCount : Nat) (stage : Phased.Stage)
    (values : (io V h outputCount stage).Inputs) : Formula.Values V h outputCount :=
  ⟨values.scalars ⟨0,by simp [io]⟩,values.scalars ⟨1,by simp [io]⟩,values.words ⟨0,by simp [io]⟩⟩

def outputs (h : HostArgs) (outputCount : Nat) (stage : Phased.Stage)
    (values : Formula.Values V h outputCount) : (io V h outputCount stage).Outputs where
  scalars := fun port => match port with
    | ⟨0,_⟩ => values.partials
    | ⟨1,_⟩ => values.output
    | ⟨n+2,hp⟩ => False.elim (by simp [io] at hp)
  words := fun port => match port with
    | ⟨0,_⟩ => values.locks
    | ⟨n+1,hp⟩ => False.elim (by simp [io] at hp)

noncomputable def mathematical (zero : V) (h : HostArgs) (outputCount : Nat) (stage : Phased.Stage)
    (values : (io V h outputCount stage).Inputs) : (io V h outputCount stage).Outputs :=
  outputs h outputCount stage (Formula.stage zero h stage (inputs h outputCount stage values))

def domain (zero : V) (h : HostArgs) (outputCount : Nat) (stage : Phased.Stage)
    (values : (io V h outputCount stage).Inputs) : Prop :=
  Formula.Domain zero h stage (inputs h outputCount stage values)

def bounds (h : HostArgs) (outputCount : Nat) : Concurrent.Bounds where
  input := fun _ => 0
  mem := fun region => if region = 0 then Formula.partialCount h else if region = 1 then outputCount else 0
  words := fun region => if region = 0 then h.numSms else 0

end VeriTile.Bench.Repairs.SpinningLock.IO

namespace VeriTile.Bench.Repairs.SpinningLock.Logical
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
noncomputable section

/-- Exact named P/C/lock snapshots of the original concurrent input windows. -/
def inputs (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    (IO.io ℝ h count stage).Inputs ≃ Formula.Values ℝ h count where
  toFun := IO.inputs h count stage
  invFun := fun values => {
    scalars := fun port => match port with
      | ⟨0,_⟩ => values.partials
      | ⟨1,_⟩ => values.output
      | ⟨n+2,hp⟩ => False.elim (by simp [IO.io] at hp)
    words := fun port => match port with
      | ⟨0,_⟩ => values.locks
      | ⟨n+1,hp⟩ => False.elim (by simp [IO.io] at hp) }
  left_inv := by
    intro values
    cases values with
    | mk scalars words =>
      apply congrArg₂ (fun s w => (⟨s,w⟩ : (IO.io ℝ h count stage).Inputs))
      · funext port
        fin_cases port <;> rfl
      · funext port
        fin_cases port <;> rfl
  right_inv := by intro values; cases values; rfl

/-- Every returned partial, output and lock cell is represented once. -/
def outputs (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    (IO.io ℝ h count stage).Outputs ≃ Formula.Values ℝ h count where
  toFun := fun values => ⟨values.scalars ⟨0,by simp [IO.io]⟩,values.scalars ⟨1,by simp [IO.io]⟩,values.words ⟨0,by simp [IO.io]⟩⟩
  invFun := IO.outputs h count stage
  left_inv := by
    intro values
    cases values with
    | mk scalars words =>
      apply congrArg₂ (fun s w => (⟨s,w⟩ : (IO.io ℝ h count stage).Outputs))
      · funext port
        fin_cases port <;> rfl
      · funext port
        fin_cases port <;> rfl
  right_inv := by intro values; cases values; rfl

def io (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    LogicalDomainIO (IO.io ℝ h count stage).Inputs (IO.io ℝ h count stage).Outputs
      (Formula.Values ℝ h count) (Formula.Values ℝ h count)
      ((IO.io ℝ h count stage).ImplementsOn (ConcurrentSource.backend realBackend)) where
  inputs := inputs h count stage
  outputs := outputs h count stage

end
end VeriTile.Bench.Repairs.SpinningLock.Logical
