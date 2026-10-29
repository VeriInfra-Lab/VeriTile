import bench.repairs.tritonbench_g.layer_norm_ops.Kernels
import VeriTile.Triton.Float.ObservedProgramIO
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public windows of the selected FP32 forward program and backward row block.
The backward weight load occurs before the loop, including for an empty loop.
Optional statistics, residual and recomputed outputs retain their source masks.
No private scratch buffer is needed by either source. -/
namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def forwardInputs (p : ForwardParameters) : List IOChannel :=
  [⟨.fp32,[p.block],p.x⟩,⟨.fp32,[p.block],p.weight⟩,
   ⟨.fp32,[p.block],p.bias⟩,⟨.fp32,[p.block],p.residual⟩]
def forwardOutputs (p : ForwardParameters) : List IOChannel :=
  [⟨.fp32,[p.block],p.y⟩,⟨.fp32,[p.block],p.residualOut⟩,
   ⟨.fp32,[],p.mean⟩,⟨.fp32,[],p.rstd⟩]
def backwardInputs (p : BackwardParameters) : List IOChannel :=
  [⟨.fp32,[p.rows,p.block],p.x⟩,⟨.fp32,[p.block],p.weight⟩,
   ⟨.fp32,[p.block],p.bias⟩,⟨.fp32,[p.rows,p.block],p.dy⟩,
   ⟨.fp32,[p.rows,p.block],p.dres⟩,⟨.fp32,[p.rows],p.mean⟩,
   ⟨.fp32,[p.rows],p.rstd⟩]
def backwardOutputs (p : BackwardParameters) : List IOChannel :=
  [⟨.fp32,[p.rows,p.block],p.dx⟩,⟨.fp32,[p.rows,p.block],p.y⟩,
   ⟨.fp32,[p.rows,p.block],p.dresIn⟩,⟨.fp32,[p.block],p.dw⟩,
   ⟨.fp32,[p.block],p.db⟩]

abbrev ForwardValues (α : Type) (p : ForwardParameters) := ChannelValues α (forwardInputs p)
abbrev BackwardValues (α : Type) (p : BackwardParameters) := ChannelValues α (backwardInputs p)

def rowStart (p : BackwardParameters) (pid : Nat) := pid*p.rowsPerProgram
def rowStop (p : BackwardParameters) (pid : Nat) := min ((pid+1)*p.rowsPerProgram) p.rows
def rowActive (p : BackwardParameters) (pid row : Nat) : Prop :=
  rowStart p pid≤row ∧ row<rowStop p pid

def forwardRead (p : ForwardParameters) (pid : Nat) : (i : Fin (forwardInputs p).length) →
    TileIndex (forwardInputs p)[i].shape → Nat
  | ⟨0,_⟩ => fun j => pid*p.xStride+j.1.val
  | ⟨1,_⟩ => fun j => j.1.val
  | ⟨2,_⟩ => fun j => j.1.val
  | ⟨3,_⟩ => fun j => pid*p.residualStride+j.1.val
  | ⟨i+4,h⟩ => False.elim (by simp [forwardInputs] at h)
def forwardInputMask (p : ForwardParameters) : (i : Fin (forwardInputs p).length) →
    TileIndex (forwardInputs p)[i].shape → Prop
  | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.1.val<p.columns
  | ⟨2,_⟩ => fun j => p.hasBias=Bool.true ∧ j.1.val<p.columns
  | ⟨3,_⟩ => fun j => p.hasResidual=Bool.true ∧ j.1.val<p.columns
  | ⟨i+4,h⟩ => False.elim (by simp [forwardInputs] at h)
def forwardWrite (p : ForwardParameters) (pid : Nat) : (o : Fin (forwardOutputs p).length) →
    TileIndex (forwardOutputs p)[o].shape → Nat
  | ⟨0,_⟩ => fun j => pid*p.yStride+j.1.val
  | ⟨1,_⟩ => fun j => pid*p.residualOutStride+j.1.val
  | ⟨2,_⟩ | ⟨3,_⟩ => fun _ => pid
  | ⟨i+4,h⟩ => False.elim (by simp [forwardOutputs] at h)
def forwardOutputMask (p : ForwardParameters) : (o : Fin (forwardOutputs p).length) →
    TileIndex (forwardOutputs p)[o].shape → Prop
  | ⟨0,_⟩ => fun j => j.1.val<p.columns
  | ⟨1,_⟩ => fun j => p.storeResidual=Bool.true ∧ j.1.val<p.columns
  | ⟨2,_⟩ => fun _ => p.rms=Bool.false
  | ⟨3,_⟩ => fun _ => True
  | ⟨i+4,h⟩ => False.elim (by simp [forwardOutputs] at h)

def backwardRead (p : BackwardParameters) : (i : Fin (backwardInputs p).length) →
    TileIndex (backwardInputs p)[i].shape → Nat
  | ⟨0,_⟩ => fun j => j.1.val*p.xStride+j.2.1.val
  | ⟨1,_⟩ | ⟨2,_⟩ => fun j => j.1.val
  | ⟨3,_⟩ => fun j => j.1.val*p.dyStride+j.2.1.val
  | ⟨4,_⟩ => fun j => j.1.val*p.dresStride+j.2.1.val
  | ⟨5,_⟩ => fun j => j.1.val*p.meanStride
  | ⟨6,_⟩ => fun j => j.1.val*p.rstdStride
  | ⟨i+7,h⟩ => False.elim (by simp [backwardInputs] at h)
def backwardInputMask (p : BackwardParameters) (pid : Nat) :
    (i : Fin (backwardInputs p).length) → TileIndex (backwardInputs p)[i].shape → Prop
  | ⟨0,_⟩ | ⟨3,_⟩ => fun j => rowActive p pid j.1.val ∧ j.2.1.val<p.columns
  | ⟨1,_⟩ => fun j => j.1.val<p.columns
  | ⟨2,_⟩ => fun j => p.recompute=Bool.true ∧ p.hasBias=Bool.true ∧ j.1.val<p.columns
  | ⟨4,_⟩ => fun j => p.hasDres=Bool.true ∧ rowActive p pid j.1.val ∧ j.2.1.val<p.columns
  | ⟨5,_⟩ => fun j => p.rms=Bool.false ∧ rowActive p pid j.1.val
  | ⟨6,_⟩ => fun j => rowActive p pid j.1.val
  | ⟨i+7,h⟩ => False.elim (by simp [backwardInputs] at h)
def backwardWrite (p : BackwardParameters) (pid : Nat) : (o : Fin (backwardOutputs p).length) →
    TileIndex (backwardOutputs p)[o].shape → Nat
  | ⟨0,_⟩ => fun j => j.1.val*p.dxStride+j.2.1.val
  | ⟨1,_⟩ => fun j => j.1.val*p.yStride+j.2.1.val
  | ⟨2,_⟩ => fun j => j.1.val*p.dresInStride+j.2.1.val
  | ⟨3,_⟩ | ⟨4,_⟩ => fun j => pid*p.columns+j.1.val
  | ⟨i+5,h⟩ => False.elim (by simp [backwardOutputs] at h)
def backwardOutputMask (p : BackwardParameters) (pid : Nat) :
    (o : Fin (backwardOutputs p).length) → TileIndex (backwardOutputs p)[o].shape → Prop
  | ⟨0,_⟩ => fun j => rowActive p pid j.1.val ∧ j.2.1.val<p.columns
  | ⟨1,_⟩ => fun j => p.recompute=Bool.true ∧ rowActive p pid j.1.val ∧ j.2.1.val<p.columns
  | ⟨2,_⟩ => fun j => p.storeDres=Bool.true ∧ rowActive p pid j.1.val ∧ j.2.1.val<p.columns
  | ⟨3,_⟩ => fun j => j.1.val<p.columns
  | ⟨4,_⟩ => fun j => p.hasBias=Bool.true ∧ j.1.val<p.columns
  | ⟨i+5,h⟩ => False.elim (by simp [backwardOutputs] at h)

def invocation {α : Type} (M : Algebra α) (source : ComputeKernel) (pid : Nat) : Invocation α where
  kernel := source
  backend := M
  pids := fun ax => if ax=0 then pid else 0

def forwardIO {α : Type} (M : Algebra α) (p : ForwardParameters) (pid : Nat) : ProgramIO α α where
  inputs := forwardInputs p
  outputs := forwardOutputs p
  scratch := []
  buffers := [p.x,p.weight,p.bias,p.residual,p.y,p.residualOut,p.mean,p.rstd]
  read := fun i _ => forwardRead p pid i
  inputMask := fun i _ => forwardInputMask p i
  write := fun o _ => forwardWrite p pid o
  outputMask := fun o _ => forwardOutputMask p o
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation M (forwardKernel p) pid]
  encode := id
  observe := fun _ _ => id

def backwardIO {α : Type} (M : Algebra α) (p : BackwardParameters) (pid : Nat) : ProgramIO α α where
  inputs := backwardInputs p
  outputs := backwardOutputs p
  scratch := []
  buffers := [p.x,p.weight,p.bias,p.dy,p.dres,p.mean,p.rstd,p.dx,p.y,p.dresIn,p.dw,p.db]
  read := fun i _ => backwardRead p i
  inputMask := fun i _ => backwardInputMask p pid i
  write := fun o _ => backwardWrite p pid o
  outputMask := fun o _ => backwardOutputMask p pid o
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation M (backwardKernel p) pid]
  encode := id
  observe := fun _ _ => id

def forwardX {α : Type} {p : ForwardParameters} (v : ForwardValues α p) (i : TileIndex [p.block]) : α :=
  v ⟨0,by simp [forwardInputs]⟩ i
def forwardWeight {α : Type} {p : ForwardParameters} (v : ForwardValues α p) (i : TileIndex [p.block]) : α :=
  v ⟨1,by simp [forwardInputs]⟩ i
def forwardBias {α : Type} {p : ForwardParameters} (v : ForwardValues α p) (i : TileIndex [p.block]) : α :=
  v ⟨2,by simp [forwardInputs]⟩ i
def forwardResidual {α : Type} {p : ForwardParameters} (v : ForwardValues α p) (i : TileIndex [p.block]) : α :=
  v ⟨3,by simp [forwardInputs]⟩ i

def forwardResult {α : Type} {p : ForwardParameters}
    (y residual : TileIndex [p.block] → α) (mean rstd : α) : ChannelValues α (forwardOutputs p)
  | ⟨0,_⟩ => y
  | ⟨1,_⟩ => residual
  | ⟨2,_⟩ => fun _ => mean
  | ⟨3,_⟩ => fun _ => rstd
  | ⟨i+4,h⟩ => False.elim (by simp [forwardOutputs] at h)

end
end VeriTile.Bench.Repairs.LayerNormOps.Interface

namespace VeriTile.Bench.Repairs.LayerNormOps.LogicalBackwardIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (p : BackwardParameters) where
  x : Fin p.rows → Fin p.block → ℝ
  weight : Fin p.block → ℝ
  bias : Fin p.block → ℝ
  dy : Fin p.rows → Fin p.block → ℝ
  dres : Fin p.rows → Fin p.block → ℝ
  mean : Fin p.rows → ℝ
  inverse : Fin p.rows → ℝ

def inputs (A : Algebra ℝ) (p : BackwardParameters) (pid : Nat) : (Interface.backwardIO A p pid).Inputs ≃ Inputs p where
  toFun := fun v => ⟨fun i0 i1 => v (0 : Fin 7) (i0,i1,()), fun i0 => v (1 : Fin 7) (i0,()), fun i0 => v (2 : Fin 7) (i0,()), fun i0 i1 => v (3 : Fin 7) (i0,i1,()), fun i0 i1 => v (4 : Fin 7) (i0,i1,()), fun i0 => v (5 : Fin 7) (i0,()), fun i0 => v (6 : Fin 7) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.x j.1 j.2.1
    | ⟨1,_⟩ => fun j => v.weight j.1
    | ⟨2,_⟩ => fun j => v.bias j.1
    | ⟨3,_⟩ => fun j => v.dy j.1 j.2.1
    | ⟨4,_⟩ => fun j => v.dres j.1 j.2.1
    | ⟨5,_⟩ => fun j => v.mean j.1
    | ⟨6,_⟩ => fun j => v.inverse j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (p : BackwardParameters) where
  dx : Fin p.rows → Fin p.block → ℝ
  y : Fin p.rows → Fin p.block → ℝ
  dres : Fin p.rows → Fin p.block → ℝ
  dw : Fin p.block → ℝ
  db : Fin p.block → ℝ

def outputs (A : Algebra ℝ) (p : BackwardParameters) (pid : Nat) : (Interface.backwardIO A p pid).Outputs ≃ Outputs p where
  toFun := fun v => ⟨fun i0 i1 => v (0 : Fin 5) (i0,i1,()), fun i0 i1 => v (1 : Fin 5) (i0,i1,()), fun i0 i1 => v (2 : Fin 5) (i0,i1,()), fun i0 => v (3 : Fin 5) (i0,()), fun i0 => v (4 : Fin 5) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.dx j.1 j.2.1
    | ⟨1,_⟩ => fun j => v.y j.1 j.2.1
    | ⟨2,_⟩ => fun j => v.dres j.1 j.2.1
    | ⟨3,_⟩ => fun j => v.dw j.1
    | ⟨4,_⟩ => fun j => v.db j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (A : Algebra ℝ) (p : BackwardParameters) (pid : Nat) : LogicalProgramIO ℝ ℝ (Inputs p) (Outputs p) where
  core := Interface.backwardIO A p pid
  inputs := inputs A p pid
  outputs := outputs A p pid


def extend {n : Nat} (x : Fin n → ℝ) (i : Nat) : ℝ := if h : i<n then x ⟨i,h⟩ else 0

def extend2 {m n : Nat} (x : Fin m → Fin n → ℝ) (i j : Nat) : ℝ :=
  if hj : j<n then if hi : i<m then x ⟨i,hi⟩ ⟨j,hj⟩ else 0 else 0

end
end VeriTile.Bench.Repairs.LayerNormOps.LogicalBackwardIO

namespace VeriTile.Bench.Repairs.LayerNormOps.LogicalForwardIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (p : ForwardParameters) where
  x : Fin p.block → ℝ
  weight : Fin p.block → ℝ
  bias : Fin p.block → ℝ
  residual : Fin p.block → ℝ

def inputs (A : Algebra ℝ) (p : ForwardParameters) (pid : Nat) : (Interface.forwardIO A p pid).Inputs ≃ Inputs p where
  toFun := fun v => ⟨fun i0 => v (0 : Fin 4) (i0,()), fun i0 => v (1 : Fin 4) (i0,()), fun i0 => v (2 : Fin 4) (i0,()), fun i0 => v (3 : Fin 4) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.x j.1
    | ⟨1,_⟩ => fun j => v.weight j.1
    | ⟨2,_⟩ => fun j => v.bias j.1
    | ⟨3,_⟩ => fun j => v.residual j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (p : ForwardParameters) where
  y : Fin p.block → ℝ
  residual : Fin p.block → ℝ
  mean : ℝ
  inverse : ℝ

def outputs (A : Algebra ℝ) (p : ForwardParameters) (pid : Nat) : (Interface.forwardIO A p pid).Outputs ≃ Outputs p where
  toFun := fun v => ⟨fun i0 => v (0 : Fin 4) (i0,()), fun i0 => v (1 : Fin 4) (i0,()), v (2 : Fin 4) (), v (3 : Fin 4) ()⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.y j.1
    | ⟨1,_⟩ => fun j => v.residual j.1
    | ⟨2,_⟩ => fun j => v.mean 
    | ⟨3,_⟩ => fun j => v.inverse 
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (A : Algebra ℝ) (p : ForwardParameters) (pid : Nat) : LogicalProgramIO ℝ ℝ (Inputs p) (Outputs p) where
  core := Interface.forwardIO A p pid
  inputs := inputs A p pid
  outputs := outputs A p pid


def extend {n : Nat} (x : Fin n → ℝ) (i : Nat) : ℝ := if h : i<n then x ⟨i,h⟩ else 0

def extend2 {m n : Nat} (x : Fin m → Fin n → ℝ) (i j : Nat) : ℝ :=
  if hj : j<n then if hi : i<m then x ⟨i,hi⟩ ⟨j,hj⟩ else 0 else 0

end
end VeriTile.Bench.Repairs.LayerNormOps.LogicalForwardIO
