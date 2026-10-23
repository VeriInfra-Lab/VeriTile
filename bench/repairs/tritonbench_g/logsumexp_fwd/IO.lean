import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.RawKernel
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO


/-! Complete row/block-grid IO for the actual FP32 source. The output channel
contains every block logsumexp value stored in the scratch tensor. Runtime
scale and per-program reduction semantics remain explicit launch data. -/
namespace VeriTile.Bench.Repairs.LogsumexpFwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (a : RawKernel.Geometry) : List IOChannel :=
  [⟨.real, [a.rows * a.columns], "input"⟩]
def outputs (a : RawKernel.Geometry) : List IOChannel :=
  [⟨.real, [a.rows * a.blocks], "scratch"⟩]
abbrev Inputs (A : Type) (a : RawKernel.Geometry) := ChannelValues A (inputs a)
abbrev Outputs (A : Type) (a : RawKernel.Geometry) := ChannelValues A (outputs a)

def inputLane (a : RawKernel.Geometry) (i : Fin (inputs a).length) :
    TileIndex (inputs a)[i].shape → Fin (a.rows * a.columns) :=
  match i with
  | ⟨0, _⟩ => fun j => j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [inputs] at h)
def outputLane (a : RawKernel.Geometry) (i : Fin (outputs a).length) :
    TileIndex (outputs a)[i].shape → Fin (a.rows * a.blocks) :=
  match i with
  | ⟨0, _⟩ => fun j => j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (a : RawKernel.Geometry) : MemoryIO A where
  inputs := inputs a
  outputs := outputs a
  scratch := []
  buffers := ["input", "scratch"]
  read := fun i _ j => (inputLane a i j).val
  inputMask := fun _ _ _ => True
  write := fun i _ j => (outputLane a i j).val
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (engine : Engine A) (a : RawKernel.Geometry) (scale : A) (j : Nat) : Invocation A where
  kernel := rowBlock "input" "scratch" a.columns a.block a.hasScale
  backend := engine.algebra
  arguments := [⟨"scale", .real, [], fun _ => scale⟩]
  pids := fun ax => if ax = 0 then j / a.blocks else if ax = 1 then j % a.blocks else 0

def programIO (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) : ProgramIO A A :=
  { memory A a with
    schedule := fun _ => jobs.map (fun j => invocation (engines j) a scale j)
    encode := id
    observe := fun _ _ => id }

def value {A : Type} {a : RawKernel.Geometry} (values : Inputs A a)
    (i : Fin (a.rows * a.columns)) : A :=
  values ⟨0, by simp [inputs]⟩ (i, PUnit.unit)

def extend {A : Type} [Inhabited A] {a : RawKernel.Geometry}
    (values : Inputs A a) (i : Nat) : A :=
  if h : i < a.rows * a.columns then value values ⟨i, h⟩ else default

def result {A : Type} {a : RawKernel.Geometry}
    (values : Fin (a.rows * a.blocks) → A) : Outputs A a :=
  fun i => match i with
  | ⟨0, _⟩ => fun j => values j.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

/-- Real inputs are ordinary real numbers embedded in the source model's
extended carrier. This is the existing mathematical scope, not a Python guard
or an empirical numerical assumption; native inputs still admit every word. -/
def RealInputs (a : RawKernel.Geometry) (values : Inputs (WithBot ℝ) a) : Prop :=
  ∀ i : Fin (a.rows * a.columns), ∃ x : ℝ, value values i = some x

def realInput {a : RawKernel.Geometry} (values : Inputs (WithBot ℝ) a) (i : Nat) : ℝ :=
  match extend values i with
  | none => 0
  | some x => x

end
end VeriTile.Bench.Repairs.LogsumexpFwd.Interface

namespace VeriTile.Bench.Repairs.LogsumexpFwd.ReaderIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Bijective dense row/column coordinates for every flat channel cell. -/
def matrix (rows columns : Nat) : Values (WithBot ℝ) .real [rows*columns] ≃
    (Fin rows → Fin columns → WithBot ℝ) where
  toFun := fun xs r c => xs (finProdFinEquiv (r,c),())
  invFun := fun xs j => let q := finProdFinEquiv.symm j.1; xs q.1 q.2
  left_inv := by
    intro xs; funext j
    change xs (finProdFinEquiv (finProdFinEquiv.symm j.1),()) = xs j
    rw [Equiv.apply_symm_apply]
  right_inv := by
    intro xs; funext r c
    change xs (finProdFinEquiv.symm (finProdFinEquiv (r,c))).1
      (finProdFinEquiv.symm (finProdFinEquiv (r,c))).2 = xs r c
    rw [Equiv.symm_apply_apply]

def inputs (a : RawKernel.Geometry) : Interface.Inputs (WithBot ℝ) a ≃
    (Fin a.rows → Fin a.columns → WithBot ℝ) :=
  (ChannelValues.singleEquiv (WithBot ℝ) _).trans (matrix a.rows a.columns)

def realInputs (a : RawKernel.Geometry) (xs : Fin a.rows → Fin a.columns → WithBot ℝ) : Prop :=
  ∀ row column, ∃ x : ℝ, xs row column = some x

def realAt {a : RawKernel.Geometry} (xs : Fin a.rows → Fin a.columns → WithBot ℝ)
    (row : Fin a.rows) (column : Nat) : ℝ :=
  if h : column<a.columns then (xs row ⟨column,h⟩).getD 0 else 0

def view (a : RawKernel.Geometry) (scale : ℝ) (jobs : List Nat) :
    LogicalProgramIO (WithBot ℝ) (WithBot ℝ)
      (Fin a.rows → Fin a.columns → WithBot ℝ) (Fin a.rows → Fin a.blocks → WithBot ℝ) where
  core := Interface.programIO (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs
  inputs := inputs a
  outputs := (ChannelValues.singleEquiv (WithBot ℝ) _).trans (matrix a.rows a.blocks)
end
end VeriTile.Bench.Repairs.LogsumexpFwd.ReaderIO
