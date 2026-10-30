import bench.tritonbench_g.quantize_kv_copy.Proofs.KernelRaw
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Tactic.FinCases

/-! Complete source IO for grouped KV quantization. Metadata is a separate
signed channel; codes and scales are separate outputs at the decoded slots. -/
namespace VeriTile.Bench.Repairs.QuantizeKvCopy.PublicIO
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
set_option autoImplicit true

def inputs (p : Parameters) (n : Nat) : List IOChannel :=
  [⟨.real, [n,p.groups,8], "K"⟩, ⟨.int, [n], "D"⟩]
def outputs (p : Parameters) (n : Nat) : List IOChannel :=
  [⟨.int, [n,p.groups,8], "O"⟩, ⟨.real, [n,p.groups], "S"⟩]
abbrev Inputs (α : Type) (p : Parameters) (n : Nat) := ChannelValues α (inputs p n)

/-- Actual source-data coordinates, extended outside the finite public windows
only to reuse the independently stated group formulas. -/
def elements [Inhabited α] {p : Parameters} {n : Nat} (values : Inputs α p n)
    (pid g c : Nat) : α :=
  if hp : pid<n then if hg : g<p.groups then if hc : c<8 then
    values ⟨0, by simp [inputs]⟩ (⟨pid,hp⟩,⟨g,hg⟩,⟨c,hc⟩,PUnit.unit)
  else default else default else default

def Metadata {α : Type} {p : Parameters} {n : Nat}
    (dest : Nat → Nat) (values : Inputs α p n) : Prop :=
  ∀ pid : Fin n, values ⟨1, by simp [inputs]⟩ (pid,PUnit.unit) = (dest pid.val : Int)

def invocation (op : Ops α) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := kernel p
  backend := backend op
  pids := fun ax => if ax=0 then pid/p.heads else pid%p.heads
  setPids := fun ax => decide (ax<2)

def sourceIO (op : Ops α) (p : Parameters) (n : Nat) (dest : Nat → Nat) : ProgramIO α α where
  inputs := inputs p n
  outputs := outputs p n
  scratch := []
  buffers := ["K","D","O","S"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun i => inputOffset p i.1.val i.2.1.val i.2.2.1.val
    | ⟨1,_⟩ => fun i => i.1.val/p.heads
    | ⟨k+2,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun i => codeOffset p (KernelGridResult.target p dest i.1.val) i.2.1.val i.2.2.1.val
    | ⟨1,_⟩ => fun i => scaleOffset p (KernelGridResult.target p dest i.1.val) i.2.1.val
    | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range n).map (invocation op p)
  encode := id
  observe := fun _ _ => id

def result {p : Parameters} {n : Nat}
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α) : ChannelValues α (outputs p n)
  | ⟨0,_⟩ => fun i => codes i.1.val i.2.1.val i.2.2.1.val
  | ⟨1,_⟩ => fun i => scales i.1.val i.2.1.val
  | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.QuantizeKvCopy.PublicIO

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.LogicalView
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section

/-- Named mathematical tensor and signed destination metadata. -/
structure Inputs (p : Parameters) (n : Nat) where
  values : Fin n → Fin p.groups → Fin (8) → ℝ
  destination : Fin n → Int

/-- Both actual outputs: signed INT8 codes and the source scale channel. -/
structure Outputs (p : Parameters) (n : Nat) where
  code : Fin n → Fin p.groups → Fin (8) → Int
  scale : Fin n → Fin p.groups → ℝ

def inputEquiv (p : Parameters) (n : Nat) : PublicIO.Inputs ℝ p n ≃ Inputs p n where
  toFun := fun xs => ⟨(fun t h c => xs ⟨0,by simp [PublicIO.inputs]⟩ (t,h,c,())),fun t => xs ⟨1,by simp [PublicIO.inputs]⟩ (t,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.values j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.destination j.1) (fun k => k.elim0))
  left_inv := by
    intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (p : Parameters) (n : Nat) : ChannelValues ℝ (PublicIO.outputs p n) ≃ Outputs p n where
  toFun := fun ys => ⟨(fun t h c => ys ⟨0,by simp [PublicIO.outputs]⟩ (t,h,c,())),fun t h => ys ⟨1,by simp [PublicIO.outputs]⟩ (t,h,())⟩
  invFun := fun ys => Fin.cases (fun j => ys.code j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => ys.scale j.1 j.2.1) (fun k => k.elim0))
  left_inv := by intro ys; funext o j; fin_cases o <;> rfl
  right_inv := by intro ys; cases ys; rfl

/-- Independent zero extension of a finite mathematical row. -/
def row {p : Parameters} {n : Nat} (xs : Inputs p n)
    (t : Fin n) (h : Fin p.groups) (c : Nat) : ℝ :=
  if hc : c<8 then xs.values t h ⟨c,hc⟩ else 0

def quantizeIO (p : Parameters) (n : Nat) (dest : Nat → Nat) :
    LogicalProgramIO ℝ ℝ (Inputs p n) (Outputs p n) where
  core := PublicIO.sourceIO KernelModels.realOps p n dest
  inputs := inputEquiv p n
  outputs := outputEquiv p n

def Metadata (dest : Nat → Nat) {p : Parameters} {n : Nat} (xs : Inputs p n) : Prop :=
  ∀ pid, xs.destination pid=(dest pid.val : Int)
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.LogicalView
