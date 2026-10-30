import bench.tritonbench_g.quantize_global.Proofs.KernelCorrectness
import VeriTile.Triton.Float.IntegerStageIO
import VeriTile.Triton.Float.IntegerProgramEquivalenceSyntax
import VeriTile.Triton.Memory.LogicalIO
import Mathlib.Tactic.FinCases

/-! Logical ports for the selected FP32 div.rn/multiply/clamp/llrint64 source.
The maximum is an explicit scalar input; the destination is signed INT8 storage
observed through the executing integer cell. -/
namespace QuantizeGlobalRepair.QuantizeIO
open VeriTile.Triton FP.Structural
noncomputable section

def ports (α : Type) (p : Parameters) (active : Fin p.n → Prop) : MemoryIO α where
  inputs := [⟨.real, [p.n], p.X⟩, ⟨.real, [], p.Maximum⟩]
  outputs := [⟨.int, [p.n], p.Out⟩]
  buffers := [p.X, p.Maximum, p.Out]
  read := by
    intro port _
    refine Fin.cases (fun i => i.1.val) (fun p1 => ?_) port
    have eq : p1 = (0 : Fin 1) := Fin.eq_zero p1
    subst p1
    exact fun _ => 0
  inputMask := fun _ _ _ => True
  write := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.1.val
  outputMask := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => active i.1
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port

/-- Independent nearest-even quantization, including zero maximum. -/
def realValues {n : Nat} (xs : TileIndex [n] → ℝ) (maximum : ℝ) : TileIndex [n] → Int :=
  fun i => Real.quantized (xs i) maximum

/-- Exact existing scalar graph: div.rn, separate multiply, clamp, signed64
rounding, then modular signed INT8 storage. All primitive inputs are retained. -/
def nativeValues {α : Type} {n : Nat} (a : Scalar α) (xs : TileIndex [n] → α)
    (maximum : α) : TileIndex [n] → Int :=
  fun i => (scalar a (xs i) maximum).toInt

/-- The declared maximum is nonnegative and dominates every input magnitude. -/
def scaleCondition {n : Nat} (xs : TileIndex [n] → ℝ) (maximum : ℝ) : Prop :=
  0 ≤ maximum ∧ ∀ i, |xs i| ≤ maximum

/-- The selected source with its unchanged integer storage and runtime marker
bindings. Every axis is installed exactly as in the existing source launch. -/
def invocation {α : Type} (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (v : QuantizeGlobalRepair.Invocation) : IntegerInvocation α where
  kernel := (kernel p.X p.Maximum p.Out p.n v.base v.config.block).program
  backend := backend rest a
  pids := fun ax => if ax = 0 then v.pid else 0
  integerStorage := QuantizeIntegerStage.byteOutput p.Out
  runtimeRegisters := QuantizeIntegerStage.outputRegister
  round := some a.round64

/-- Complete selected grid, including every supplied autotune trial. -/
def io {α : Type} (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (trials : Nat → List Config) : IntegerProgramIO α α :=
  { ports α p (fun _ => True) with
    schedule := fun _ => (schedule p.n trials).map (invocation rest a p)
    encode := id
    observe := fun _ _ => id }

/-- A second sequence of the existing configurations on the identical ports. -/
def variant {α : Type} (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (left right : Nat → List Config) : (io rest a p left).Variant where
  schedule := fun _ => (schedule p.n right).map (invocation rest a p)
  observe := fun _ _ => id

/-- Actual complete left configuration schedule, with the same typed ports,
input encoding and unrestricted native word domain as the existing pair. -/
def original {α : Type} (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (trials : Nat → List Config) :
    IntegerProgramIO.Implementation (io rest a p trials) (fun _ => True) :=
  IntegerProgramIO.Implementation.original (io rest a p trials) (fun _ => True)

/-- The existing right configuration schedule on precisely the left IO. -/
def candidate {α : Type} (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (left right : Nat → List Config) :
    IntegerProgramIO.Implementation (io rest a p left) (fun _ => True) :=
  IntegerProgramIO.Implementation.fromVariant (fun _ => True) (variant rest a p left right)

def target {α : Type} (rest : Algebra α) (a : Scalar α) (p : Parameters)
    (trials : Nat → List Config)
    (f : (TileIndex [p.n] → α) → α → TileIndex [p.n] → Int) :
    (io rest a p trials).Inputs → (io rest a p trials).Outputs := by
  intro vals port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  exact f (vals (0 : Fin 2)) (vals (1 : Fin 2) ())

def domain (rest : Algebra ℝ) (p : Parameters) (trials : Nat → List Config)
    (vals : (io rest Real.arithmetic p trials).Inputs) : Prop :=
  scaleCondition (vals (0 : Fin 2)) (vals (1 : Fin 2) ())

end
end QuantizeGlobalRepair.QuantizeIO

namespace QuantizeGlobalRepair.LogicalView
open VeriTile.Triton FP.Structural QuantizeIO
noncomputable section
structure Inputs (n : Nat) where
  values : Fin n → ℝ
  maximum : ℝ

def inputEquiv (rest : Algebra ℝ) (p : Parameters) (trials : Nat → List Config) :
    (io rest Real.arithmetic p trials).Inputs ≃ Inputs p.n where
  toFun := fun xs => ⟨fun i => xs ⟨0,by simp [QuantizeIO.io,QuantizeIO.ports]⟩ (i,()),xs ⟨1,by simp [QuantizeIO.io,QuantizeIO.ports]⟩ ()⟩
  invFun := fun xs => Fin.cases (fun j => xs.values j.1)
    (Fin.cases (fun _ => xs.maximum) (fun k => k.elim0))
  left_inv := by intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (rest : Algebra ℝ) (p : Parameters) (trials : Nat → List Config) :
    (io rest Real.arithmetic p trials).Outputs ≃ (Fin p.n → Int) where
  toFun := fun ys i => ys ⟨0,by simp [QuantizeIO.io,QuantizeIO.ports]⟩ (i,())
  invFun := fun ys => Fin.cases (fun j => ys j.1) (fun k => k.elim0)
  left_inv := by intro ys; funext o j; fin_cases o; rfl
  right_inv := fun _ => rfl

def quantizeIO (rest : Algebra ℝ) (p : Parameters) (trials : Nat → List Config) :
    LogicalDomainIO (io rest Real.arithmetic p trials).Inputs (io rest Real.arithmetic p trials).Outputs
      (Inputs p.n) (Fin p.n → Int) (io rest Real.arithmetic p trials).ImplementsOn where
  inputs := inputEquiv rest p trials
  outputs := outputEquiv rest p trials
end
end QuantizeGlobalRepair.LogicalView
