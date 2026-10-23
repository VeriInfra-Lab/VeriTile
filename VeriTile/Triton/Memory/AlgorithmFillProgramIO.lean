import VeriTile.Triton.Memory.AlgorithmProgramIO

/-! Finite initialization between existing Algorithm source stages.

The only additional operation is a constant typed-cell fill of a contiguous
regional interval. It has no user-supplied executor, safety predicate, or
state transformer. Source stages retain their existing execution semantics.
-/
namespace VeriTile.Triton
noncomputable section

structure AlgorithmFill where
  region : RegionName
  start : Nat
  count : Nat
  value : MemCell

namespace AlgorithmFill
def Contains (fill : AlgorithmFill) (r : RegionName) (address : Nat) : Prop :=
  r=fill.region ∧ fill.start ≤ address ∧ address<fill.start+fill.count

instance (fill : AlgorithmFill) (r : RegionName) (address : Nat) : Decidable (fill.Contains r address) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def apply (fill : AlgorithmFill) (s : BlockState) : BlockState :=
  {s with mem:=fun r address=>if fill.Contains r address then fill.value else s.mem r address}

def Safe (fill : AlgorithmFill) (bounds : RegionBounds) : Prop :=
  ∀ i, i<fill.count → fill.start+i<bounds fill.region

@[simp] theorem apply_mem_inside (fill : AlgorithmFill) (s : BlockState)
    (i : Nat) (hi : i<fill.count) :
    (fill.apply s).mem fill.region (fill.start+i)=fill.value := by
  simp [apply,Contains,hi]

theorem apply_mem_outside (fill : AlgorithmFill) (s : BlockState)
    (r : RegionName) (address : Nat) (outside : ¬fill.Contains r address) :
    (fill.apply s).mem r address=s.mem r address := by simp [apply,outside]

@[simp] theorem apply_zero (region : RegionName) (start : Nat) (value : MemCell)
    (s : BlockState) : (AlgorithmFill.mk region start 0 value).apply s=s := by
  have empty (r : RegionName) (address : Nat) :
      ¬(AlgorithmFill.mk region start 0 value).Contains r address := by
    simp only [Contains,Nat.add_zero]
    omega
  simp [apply,empty]
end AlgorithmFill

inductive AlgorithmFillStage where
  | source (stage : AlgorithmStage)
  | fill (initialization : AlgorithmFill)

namespace AlgorithmFillStage
def Runs : AlgorithmFillStage → BlockState → BlockState → Prop
  | .source stage,s,t => stage.Runs s t
  | .fill initialization,s,t => t=initialization.apply s

def Safe (bounds : RegionBounds) : AlgorithmFillStage → BlockState → Prop
  | .source stage,s => stage.Safe bounds s
  | .fill initialization,_ => initialization.Safe bounds
end AlgorithmFillStage

inductive AlgorithmFillExecution : List AlgorithmFillStage → BlockState → BlockState → Type where
  | nil (s : BlockState) : AlgorithmFillExecution [] s s
  | cons {stage : AlgorithmFillStage} {rest : List AlgorithmFillStage} {s middle final : BlockState}
      (head : stage.Runs s middle) (tail : AlgorithmFillExecution rest middle final) :
      AlgorithmFillExecution (stage::rest) s final

namespace AlgorithmFillExecution
def Safe {stages : List AlgorithmFillStage} {s t : BlockState}
    (trace : AlgorithmFillExecution stages s t) (bounds : RegionBounds) : Prop :=
  match trace with
  | .nil _ => True
  | @cons stage _ initial _ _ _ tail => stage.Safe bounds initial ∧ tail.Safe bounds

def ofSource {stages : List AlgorithmStage} {s t : BlockState}
    (trace : AlgorithmExecution stages s t) :
    AlgorithmFillExecution (stages.map AlgorithmFillStage.source) s t := by
  induction trace with
  | nil s => exact .nil s
  | cons head tail ih => exact .cons head ih

theorem ofSource_safe {stages : List AlgorithmStage} {s t : BlockState}
    (trace : AlgorithmExecution stages s t) (bounds : RegionBounds) :
    (ofSource trace).Safe bounds ↔ trace.Safe bounds := by
  induction trace with
  | nil s => rfl
  | cons head tail ih => exact and_congr Iff.rfl ih
end AlgorithmFillExecution

structure AlgorithmFillProgramIO extends AlgorithmMemoryIO where
  initialUndef : AlgorithmInitialUndef := .preserve
  stages : (∀ k, Fin (iarity k) → (ity k).carrier) → List AlgorithmFillStage

namespace AlgorithmFillProgramIO
abbrev Ctx (io : AlgorithmFillProgramIO) := io.toAlgorithmMemoryIO.Ctx
abbrev InputBinding (io : AlgorithmFillProgramIO) := io.toAlgorithmMemoryIO.InputBinding
abbrev InBounds (io : AlgorithmFillProgramIO) := io.toAlgorithmMemoryIO.InBounds
abbrev OutputBinding (io : AlgorithmFillProgramIO) := io.toAlgorithmMemoryIO.OutputBinding
abbrev Frame (io : AlgorithmFillProgramIO) := io.toAlgorithmMemoryIO.Frame

/-- Execution, outputs and exact exterior cells are unconditional on bounds.
Declared finite bounds imply safety of that same source/fill execution. -/
def ImplementsOn3D (io : AlgorithmFillProgramIO) (domain : io.Ctx → Prop)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ vals,domain vals → ∀ initial,io.InputBinding initial vals →
    ∃ final,∃ trace : AlgorithmFillExecution (io.stages vals) (io.initialUndef.initial initial) final,
      (∀ bounds,io.InBounds bounds initial vals → trace.Safe bounds) ∧
      io.OutputBinding initial final vals (f (initial.pids 0) (initial.pids 1) (initial.pids 2) vals) ∧
      io.Frame initial final vals

def ImplementsOn (io : AlgorithmFillProgramIO) (domain : io.Ctx → Prop)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn3D domain (fun _ _ _=>f)

def Implements (io : AlgorithmFillProgramIO)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn (fun _=>True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f
end AlgorithmFillProgramIO
end
end VeriTile.Triton
