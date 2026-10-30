import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelModels
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.LogicalProgramIO

/-! Full-grid IO for the actual signed INT8-to-FP32 dequantization source.
The scale input has its own N-element window; the complete K-by-N output is
observed, including singleton and empty dimensions admitted by the raw model. -/
namespace VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

def inputs (p : Parameters) : List IOChannel :=
  [⟨.int, [p.K*p.N], p.b⟩, ⟨.real, [p.N], p.scale⟩]
def outputs (p : Parameters) : List IOChannel := [⟨.real, [p.K*p.N], p.output⟩]
abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (inputs p)

def quantized {α : Type} {p : Parameters} (values : Inputs α p)
    (i : Fin (p.K*p.N)) : Int := values ⟨0, by simp [inputs]⟩ (i, PUnit.unit)
def scale {α : Type} {p : Parameters} (values : Inputs α p) (i : Fin p.N) : α :=
  values ⟨1, by simp [inputs]⟩ (i, PUnit.unit)

def column (p : Parameters) (i : Fin (p.K*p.N)) : Fin p.N :=
  ⟨i.val % p.N, Nat.mod_lt _ (by have h := i.isLt; nlinarith)⟩

def SignedBytes {α : Type} {p : Parameters} (values : Inputs α p) : Prop :=
  ∀ i, ∃ word : BitVec 8, quantized values i = word.toInt

def invocation {α : Type} (M : Algebra α) (p : Parameters) (version : Bool)
    (pid : KernelResult.Program) : Invocation α where
  kernel := kernel version p
  backend := M
  pids := fun ax => if ax = 0 then pid.1 else pid.2
  setPids := fun ax => decide (ax < 2)

def sourceIO {α : Type} (M : Algebra α) (p : Parameters) (version : Bool) : ProgramIO α α where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.b, p.scale, p.output]
  read := fun port _ => match port with
    | ⟨0, _⟩ => fun i => i.1.val
    | ⟨1, _⟩ => fun i => i.1.val
    | ⟨k+2, h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0, _⟩ => fun i => i.1.val
    | ⟨k+1, h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (KernelGrid.order p).map (invocation M p version)
  encode := id
  observe := fun _ _ => id

def variant {α : Type} (M : Algebra α) (p : Parameters) :
    ProgramIO.Variant (sourceIO M p Bool.false) where
  schedule := fun _ => (KernelGrid.order p).map (invocation M p Bool.true)
  observe := fun _ _ => id

def result {α : Type} {p : Parameters} (f : Int → α → α) (values : Inputs α p) :
    ChannelValues α (outputs p)
  | ⟨0, _⟩ => fun i => f (quantized values i.1) (scale values (column p i.1))
  | ⟨k+1, h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO

namespace VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

structure LogicalInputs (α : Type) (p : Parameters) where
  weights : Fin p.K → Fin p.N → Int
  scales : Fin p.N → α
abbrev LogicalOutputs (α : Type) (p : Parameters) := Fin p.K → Fin p.N → α

def inputEquiv (α : Type) (p : Parameters) : Inputs α p ≃ LogicalInputs α p where
  toFun := fun values => ⟨fun row col => quantized values (finProdFinEquiv (row,col)), scale values⟩
  invFun := fun xs => Fin.cases
    (fun j => xs.weights (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2)
    (Fin.cases (fun j => xs.scales j.1) (fun o => Fin.elim0 o))
  left_inv := by
    intro values
    funext o j
    fin_cases o
    · rcases j with ⟨i,u⟩
      cases u
      change quantized values (finProdFinEquiv (finProdFinEquiv.symm i)) = _
      rw [Equiv.apply_symm_apply]
      rfl
    · rcases j with ⟨i,u⟩
      cases u
      rfl
  right_inv := by
    intro xs
    cases xs with
    | mk weights scales =>
      change (⟨fun row col => weights (finProdFinEquiv.symm (finProdFinEquiv (row,col))).1
        (finProdFinEquiv.symm (finProdFinEquiv (row,col))).2, scales⟩ : LogicalInputs α p) = ⟨weights,scales⟩
      congr 1
      funext row col
      change weights (finProdFinEquiv.symm (finProdFinEquiv (row,col))).1
        (finProdFinEquiv.symm (finProdFinEquiv (row,col))).2 = _
      rw [Equiv.symm_apply_apply]

def outputEquiv (α : Type) (p : Parameters) :
    ChannelValues α (outputs p) ≃ LogicalOutputs α p where
  toFun := fun values row col => values ⟨0,by simp [outputs]⟩ (finProdFinEquiv (row,col),())
  invFun := fun xs => Fin.cases
    (fun j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2)
    (fun o => Fin.elim0 o)
  left_inv := by
    intro values
    funext o j
    fin_cases o
    rcases j with ⟨i,u⟩
    cases u
    change values ⟨0,by simp [outputs]⟩ (finProdFinEquiv (finProdFinEquiv.symm i),()) = _
    rw [Equiv.apply_symm_apply]
    rfl
  right_inv := by
    intro xs
    funext row col
    change xs (finProdFinEquiv.symm (finProdFinEquiv (row,col))).1
      (finProdFinEquiv.symm (finProdFinEquiv (row,col))).2 = _
    rw [Equiv.symm_apply_apply]

def LogicalSignedBytes (xs : LogicalInputs α p) : Prop :=
  ∀ row col, ∃ word : BitVec 8, xs.weights row col = word.toInt

def logicalIO (rest : Algebra ℝ) (p : Parameters) :
    LogicalProgramIO ℝ ℝ (LogicalInputs ℝ p) (LogicalOutputs ℝ p) where
  core := sourceIO (KernelModels.realAlgebra rest) p Bool.false
  inputs := inputEquiv ℝ p
  outputs := outputEquiv ℝ p

end
end VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO
