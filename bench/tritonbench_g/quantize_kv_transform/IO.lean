import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelInterface
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Tactic.FinCases
import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelRealModel

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.IO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

abbrev channels (p : Parameters) (count : Nat) : List IOChannel :=
  [⟨.real, [count, p.heads, p.columns], "input"⟩, ⟨.int, [count], "destination"⟩]
abbrev Inputs (α : Type) (p : Parameters) (count : Nat) := ChannelValues α (channels p count)

def values {α : Type} [Inhabited α] {p : Parameters} {count : Nat}
    (vals : Inputs α p count) (token head column : Nat) : α :=
  if h : token < count ∧ head < p.heads ∧ column < p.columns then
    vals 0 (⟨token, h.1⟩, ⟨head, h.2.1⟩, ⟨column, h.2.2⟩, PUnit.unit) else default

def destination {α : Type} {p : Parameters} {count : Nat}
    (vals : Inputs α p count) (token : Nat) : Nat :=
  if h : token < count then (vals 1 (⟨token, h⟩, PUnit.unit)).toNat else 0

/-- Existing metadata conditions: nonnegative destination indices, injective
on active tokens. Allocation bounds are supplied by the shared IO judgment. -/
def Destinations {α : Type} (p : Parameters) (count : Nat) (vals : Inputs α p count) : Prop :=
  (∀ token : Fin count, (0 : Int) ≤ (vals 1 (token, PUnit.unit) : Int)) ∧
  (∀ x y, x < count → y < count → destination vals x = destination vals y → x = y)

def launch {α : Type} (A : Algebra α) (p : Parameters) (token : Nat) : Invocation α where
  kernel := kernel p
  backend := A
  pids := fun _ => token
  setPids := fun ax => decide (ax = 0)

def schedule {α : Type} (A : Algebra α) (p : Parameters) (order : List Nat) :=
  order.map (launch A p)

/-- Full token grid; INT8 codes and FP32 scales scatter to caller destinations.
Both outputs are explicitly indexed by their source token/head/column. -/
abbrev quantizeIO {α : Type} (A : Algebra α) (p : Parameters) (count : Nat)
    (order : List Nat) : ProgramIO α α where
  inputs := channels p count
  outputs := [⟨.int, [count, p.heads, p.columns], "output"⟩,
    ⟨.real, [count, p.heads], "scale"⟩]
  scratch := []
  buffers := ["input", "destination", "output", "scale"]
  read := Fin.cases
    (fun _ j => j.1.val * (p.heads * p.columns) + j.2.1.val * p.columns + j.2.2.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => i.elim0))
  inputMask := fun _ _ _ => True
  write := Fin.cases
    (fun vals j => destination vals j.1.val * (p.heads * p.columns) + p.columns * j.2.1.val + j.2.2.1.val)
    (Fin.cases (fun vals j => destination vals j.1.val * p.heads + j.2.1.val) (fun i => i.elim0))
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => schedule A p order
  encode := id
  observe := fun _ _ => id

def outputValues {α : Type} {p : Parameters} {count : Nat}
    (code : Fin count → Fin p.heads → Fin p.columns → Int)
    (scale : Fin count → Fin p.heads → α) :
    ChannelValues α [⟨.int, [count, p.heads, p.columns], "output"⟩,
      ⟨.real, [count, p.heads], "scale"⟩] :=
  Fin.cases (fun j => code j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => scale j.1 j.2.1) (fun i => i.elim0))
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.IO

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.LogicalView
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

/-- Named mathematical tensor and signed destination metadata. -/
structure Inputs (p : Parameters) (count : Nat) where
  values : Fin count → Fin p.heads → Fin (p.columns) → ℝ
  destination : Fin count → Int

/-- Both actual outputs: signed INT8 codes and the source scale channel. -/
structure Outputs (p : Parameters) (count : Nat) where
  code : Fin count → Fin p.heads → Fin (p.columns) → Int
  scale : Fin count → Fin p.heads → ℝ

def inputEquiv (p : Parameters) (count : Nat) : IO.Inputs ℝ p count ≃ Inputs p count where
  toFun := fun xs => ⟨(fun t h c => xs ⟨0,by simp [IO.channels]⟩ (t,h,c,())),fun t => xs ⟨1,by simp [IO.channels]⟩ (t,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.values j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.destination j.1) (fun k => k.elim0))
  left_inv := by
    intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (p : Parameters) (count : Nat) : ChannelValues ℝ ([⟨.int, [count, p.heads, p.columns], "output"⟩,⟨.real, [count, p.heads], "scale"⟩]) ≃ Outputs p count where
  toFun := fun ys => ⟨(fun t h c => ys ⟨0,by simp⟩ (t,h,c,())),fun t h => ys ⟨1,by simp⟩ (t,h,())⟩
  invFun := fun ys => Fin.cases (fun j => ys.code j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => ys.scale j.1 j.2.1) (fun k => k.elim0))
  left_inv := by intro ys; funext o j; fin_cases o <;> rfl
  right_inv := by intro ys; cases ys; rfl

/-- Independent zero extension of a finite mathematical row. -/
def row {p : Parameters} {count : Nat} (xs : Inputs p count)
    (t : Fin count) (h : Fin p.heads) (c : Nat) : ℝ :=
  if hc : c<p.columns then xs.values t h ⟨c,hc⟩ else 0

def quantizeIO (rest : Algebra ℝ) (p : Parameters) (count : Nat) (order : List Nat) :
    LogicalProgramIO ℝ ℝ (Inputs p count) (Outputs p count) where
  core := IO.quantizeIO (KernelExecution.backend (KernelRealModel.algebra rest)
    KernelRealModel.convert KernelRealModel.equal) p count order
  inputs := inputEquiv p count
  outputs := outputEquiv p count

def Destinations {p : Parameters} {count : Nat} (xs : Inputs p count) : Prop :=
  (∀ t, 0≤xs.destination t) ∧
  (∀ a b, (xs.destination a).toNat=(xs.destination b).toNat → a=b)
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.LogicalView
