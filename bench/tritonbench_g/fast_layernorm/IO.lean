import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Logical IO of the three actual source kernels. Each interface describes one
program, matching the established theorem scope. Inputs are masked exactly at
source loads; the affine scalar statistics are read for every loop row even
when no column is active. Outputs have their separate source masks. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def forwardInputs (B : Nat) : List IOChannel :=
  [⟨.real,[B],"x"⟩,⟨.real,[B],"w"⟩,⟨.real,[B],"b"⟩]
def forwardOutputs (B : Nat) : List IOChannel :=
  [⟨.real,[B],"y"⟩,⟨.real,[],"r"⟩,⟨.real,[],"mu"⟩]
def backwardInputs (B : Nat) : List IOChannel :=
  [⟨.real,[B],"x"⟩,⟨.real,[B],"dy"⟩,⟨.real,[B],"w"⟩,⟨.real,[],"mu"⟩,⟨.real,[],"r"⟩]
def backwardOutputs (B : Nat) : List IOChannel := [⟨.real,[B],"dx"⟩]
def affineInputs (rows : Nat) : List IOChannel :=
  [⟨.real,[rows,128],"x"⟩,⟨.real,[rows,128],"dy"⟩,⟨.real,[rows],"mu"⟩,⟨.real,[rows],"r"⟩]
def affineOutputs : List IOChannel := [⟨.real,[128],"dw"⟩,⟨.real,[128],"db"⟩]

abbrev ForwardValues (α : Type) (B : Nat) := ChannelValues α (forwardInputs B)
abbrev BackwardValues (α : Type) (B : Nat) := ChannelValues α (backwardInputs B)
abbrev AffineValues (α : Type) (rows : Nat) := ChannelValues α (affineInputs rows)

def forwardRead (N B pid : Nat) : (i : Fin (forwardInputs B).length) →
    TileIndex (forwardInputs B)[i].shape → Nat
  | ⟨0,_⟩ => fun j => pid*N+j.1.val
  | ⟨1,_⟩ => fun j => j.1.val
  | ⟨2,_⟩ => fun j => j.1.val
  | ⟨i+3,h⟩ => False.elim (by simp [forwardInputs] at h)
def forwardInputMask (N B : Nat) (bias : Bool) : (i : Fin (forwardInputs B).length) →
    TileIndex (forwardInputs B)[i].shape → Prop
  | ⟨0,_⟩ => fun j => j.1.val<N
  | ⟨1,_⟩ => fun j => j.1.val<N
  | ⟨2,_⟩ => fun j => bias=Bool.true ∧ j.1.val<N
  | ⟨i+3,h⟩ => False.elim (by simp [forwardInputs] at h)
def forwardWrite (N B pid : Nat) : (i : Fin (forwardOutputs B).length) →
    TileIndex (forwardOutputs B)[i].shape → Nat
  | ⟨0,_⟩ => fun j => pid*N+j.1.val
  | ⟨1,_⟩ => fun _ => pid
  | ⟨2,_⟩ => fun _ => pid
  | ⟨i+3,h⟩ => False.elim (by simp [forwardOutputs] at h)
def forwardOutputMask (N B : Nat) : (i : Fin (forwardOutputs B).length) →
    TileIndex (forwardOutputs B)[i].shape → Prop
  | ⟨0,_⟩ => fun j => j.1.val<N
  | ⟨1,_⟩ => fun _ => True
  | ⟨2,_⟩ => fun _ => True
  | ⟨i+3,h⟩ => False.elim (by simp [forwardOutputs] at h)

def backwardRead (N B pid : Nat) : (i : Fin (backwardInputs B).length) →
    TileIndex (backwardInputs B)[i].shape → Nat
  | ⟨0,_⟩ => fun j => pid*N+j.1.val
  | ⟨1,_⟩ => fun j => pid*N+j.1.val
  | ⟨2,_⟩ => fun j => j.1.val
  | ⟨3,_⟩ => fun _ => pid
  | ⟨4,_⟩ => fun _ => pid
  | ⟨i+5,h⟩ => False.elim (by simp [backwardInputs] at h)
def backwardInputMask (N B : Nat) : (i : Fin (backwardInputs B).length) →
    TileIndex (backwardInputs B)[i].shape → Prop
  | ⟨0,_⟩ => fun j => j.1.val<N
  | ⟨1,_⟩ => fun j => j.1.val<N
  | ⟨2,_⟩ => fun j => j.1.val<N
  | ⟨3,_⟩ => fun _ => True
  | ⟨4,_⟩ => fun _ => True
  | ⟨i+5,h⟩ => False.elim (by simp [backwardInputs] at h)
def backwardIndex (B : Nat) : (i : Fin (backwardOutputs B).length) →
    TileIndex (backwardOutputs B)[i].shape → Fin B
  | ⟨0,_⟩ => fun j => j.1
  | ⟨i+1,h⟩ => False.elim (by simp [backwardOutputs] at h)

def affineRead (rows N pid : Nat) : (i : Fin (affineInputs rows).length) →
    TileIndex (affineInputs rows)[i].shape → Nat
  | ⟨0,_⟩ => fun j => j.1.val*N+(pid*128+j.2.1.val)
  | ⟨1,_⟩ => fun j => j.1.val*N+(pid*128+j.2.1.val)
  | ⟨2,_⟩ => fun j => j.1.val
  | ⟨3,_⟩ => fun j => j.1.val
  | ⟨i+4,h⟩ => False.elim (by simp [affineInputs] at h)
def affineInputMask (rows N pid : Nat) : (i : Fin (affineInputs rows).length) →
    TileIndex (affineInputs rows)[i].shape → Prop
  | ⟨0,_⟩ => fun j => pid*128+j.2.1.val<N
  | ⟨1,_⟩ => fun j => pid*128+j.2.1.val<N
  | ⟨2,_⟩ => fun _ => True
  | ⟨3,_⟩ => fun _ => True
  | ⟨i+4,h⟩ => False.elim (by simp [affineInputs] at h)
def affineIndex : (i : Fin affineOutputs.length) → TileIndex affineOutputs[i].shape → Fin 128
  | ⟨0,_⟩ => fun j => j.1
  | ⟨1,_⟩ => fun j => j.1
  | ⟨i+2,h⟩ => False.elim (by simp [affineOutputs] at h)
def affineOutputMask (wc bc pid : Nat) (bias : Bool) : (i : Fin affineOutputs.length) →
    TileIndex affineOutputs[i].shape → Prop
  | ⟨0,_⟩ => fun j => pid*128+j.1.val<wc
  | ⟨1,_⟩ => fun j => bias=Bool.true ∧ pid*128+j.1.val<bc
  | ⟨i+2,h⟩ => False.elim (by simp [affineOutputs] at h)

def invocation {α : Type} (M : Algebra α) (kernel : ComputeKernel) (pid : Nat) : Invocation α where
  kernel := kernel
  backend := engine M
  pids := fun ax => if ax=0 then pid else 0

def forwardIO {α : Type} (M : Algebra α) (N B pid : Nat) (eps : ℝ) (bias : Bool) : ProgramIO α α where
  inputs := forwardInputs B
  outputs := forwardOutputs B
  scratch := []
  buffers := ["x","w","b","y","r","mu"]
  read := fun i _ => forwardRead N B pid i
  inputMask := fun i _ => forwardInputMask N B bias i
  write := fun o _ => forwardWrite N B pid o
  outputMask := fun o _ => forwardOutputMask N B o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [invocation M (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) pid]
  encode := id
  observe := fun _ _ => id

def backwardIO {α : Type} (M : Algebra α) (N B pid : Nat) : ProgramIO α α where
  inputs := backwardInputs B
  outputs := backwardOutputs B
  scratch := []
  buffers := ["x","dy","w","mu","r","dx"]
  read := fun i _ => backwardRead N B pid i
  inputMask := fun i _ => backwardInputMask N B i
  write := fun o _ j => pid*N+(backwardIndex B o j).val
  outputMask := fun o _ j => (backwardIndex B o j).val<N
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [invocation M (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) pid]
  encode := id
  observe := fun _ _ => id

def affineIO {α : Type} (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) : ProgramIO α α where
  inputs := affineInputs rows
  outputs := affineOutputs
  scratch := []
  buffers := ["x","dy","mu","r","dw","db"]
  read := fun i _ => affineRead rows N pid i
  inputMask := fun i _ => affineInputMask rows N pid i
  write := fun o _ j => pid*128+(affineIndex o j).val
  outputMask := fun o _ => affineOutputMask wc bc pid bias o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [invocation M (affineKernel "dw" "db" "dy" "x" "r" "mu" rows N wc bc bias) pid]
  encode := id
  observe := fun _ _ => id

def forwardX {α : Type} {B : Nat} (v : ForwardValues α B) (j : Fin B) : α := v ⟨0,by simp [forwardInputs]⟩ (j,PUnit.unit)
def forwardW {α : Type} {B : Nat} (v : ForwardValues α B) (j : Fin B) : α := v ⟨1,by simp [forwardInputs]⟩ (j,PUnit.unit)
def forwardB {α : Type} {B : Nat} (v : ForwardValues α B) (j : Fin B) : α := v ⟨2,by simp [forwardInputs]⟩ (j,PUnit.unit)
def backwardX {α : Type} {B : Nat} (v : BackwardValues α B) (j : Fin B) : α := v ⟨0,by simp [backwardInputs]⟩ (j,PUnit.unit)
def backwardDY {α : Type} {B : Nat} (v : BackwardValues α B) (j : Fin B) : α := v ⟨1,by simp [backwardInputs]⟩ (j,PUnit.unit)
def backwardW {α : Type} {B : Nat} (v : BackwardValues α B) (j : Fin B) : α := v ⟨2,by simp [backwardInputs]⟩ (j,PUnit.unit)
def backwardMean {α : Type} {B : Nat} (v : BackwardValues α B) : α := v ⟨3,by simp [backwardInputs]⟩ PUnit.unit
def backwardInverse {α : Type} {B : Nat} (v : BackwardValues α B) : α := v ⟨4,by simp [backwardInputs]⟩ PUnit.unit

def affineX {α : Type} {rows : Nat} (v : AffineValues α rows) (r : Fin rows) (j : Fin 128) : α := v ⟨0,by simp [affineInputs]⟩ (r,j,PUnit.unit)
def affineDY {α : Type} {rows : Nat} (v : AffineValues α rows) (r : Fin rows) (j : Fin 128) : α := v ⟨1,by simp [affineInputs]⟩ (r,j,PUnit.unit)
def affineMean {α : Type} {rows : Nat} (v : AffineValues α rows) (r : Fin rows) : α := v ⟨2,by simp [affineInputs]⟩ (r,PUnit.unit)
def affineInverse {α : Type} {rows : Nat} (v : AffineValues α rows) (r : Fin rows) : α := v ⟨3,by simp [affineInputs]⟩ (r,PUnit.unit)

def extend {α : Type} [Inhabited α] {n : Nat} (x : Fin n → α) (i : Nat) : α :=
  if h : i<n then x ⟨i,h⟩ else default
@[simp] theorem extend_fin {α : Type} [Inhabited α] {n : Nat} (x : Fin n → α) (i : Fin n) :
    extend x i.val=x i := by simp [extend]

def forwardResult {α : Type} {B : Nat} (y : Fin B → α) (inv avg : α) : ChannelValues α (forwardOutputs B)
  | ⟨0,_⟩ => fun j => y j.1
  | ⟨1,_⟩ => fun _ => inv
  | ⟨2,_⟩ => fun _ => avg
  | ⟨i+3,h⟩ => False.elim (by simp [forwardOutputs] at h)
def backwardResult {α : Type} {B : Nat} (dx : Fin B → α) : ChannelValues α (backwardOutputs B)
  | ⟨0,_⟩ => fun j => dx j.1
  | ⟨i+1,h⟩ => False.elim (by simp [backwardOutputs] at h)
def affineResult {α : Type} (dw db : Fin 128 → α) : ChannelValues α affineOutputs
  | ⟨0,_⟩ => fun j => dw j.1
  | ⟨1,_⟩ => fun j => db j.1
  | ⟨i+2,h⟩ => False.elim (by simp [affineOutputs] at h)
end
end VeriTile.Bench.Repairs.FastLayernorm.Interface

namespace VeriTile.Bench.Repairs.FastLayernorm.IOPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite Interface
noncomputable section
variable {α : Type} [Inhabited α]

inductive Stage where
  | forward (N B pid : Nat) (eps : ℝ) (bias : Bool)
  | backward (N B pid : Nat)
  | affine (rows N wc bc pid : Nat) (bias : Bool)

def io (M : Algebra α) : Stage → ProgramIO α α
  | .forward N B pid eps bias => forwardIO M N B pid eps bias
  | .backward N B pid => backwardIO M N B pid
  | .affine rows N wc bc pid bias => affineIO M rows N wc bc pid bias

def variant (M : Algebra α) (stage : Stage) : ProgramIO.Variant (io M stage) :=
  match stage with
  | .forward N B pid eps bias =>
    { schedule := fun _ => [invocation M (forwardIndexCommuted "y" "x" "w" "b" "r" "mu" N B eps bias) pid]
      observe := fun _ _ => id }
  | .backward N B pid =>
    { schedule := fun _ => [invocation M (backwardIndexCommuted "dx" "dy" "x" "w" "r" "mu" N B) pid]
      observe := fun _ _ => id }
  | .affine rows N wc bc pid bias =>
    { schedule := fun _ => [invocation M (affineDirectOffsets "dw" "db" "dy" "x" "r" "mu" rows N wc bc bias) pid]
      observe := fun _ _ => id }
end
end VeriTile.Bench.Repairs.FastLayernorm.IOPair

namespace VeriTile.Bench.Repairs.FastLayernorm.LogicalIO
open VeriTile Triton FP.Structural
noncomputable section

structure ForwardInputs (B : Nat) where
  x : Fin B → ℝ
  weight : Fin B → ℝ
  bias : Fin B → ℝ
structure ForwardOutputs (B : Nat) where
  values : Fin B → ℝ
  inverse : ℝ
  mean : ℝ
structure BackwardInputs (B : Nat) where
  x : Fin B → ℝ
  upstream : Fin B → ℝ
  weight : Fin B → ℝ
  mean : ℝ
  inverse : ℝ
structure AffineInputs (rows : Nat) where
  x : Fin rows → Fin 128 → ℝ
  upstream : Fin rows → Fin 128 → ℝ
  mean : Fin rows → ℝ
  inverse : Fin rows → ℝ
structure AffineOutputs where
  weight : Fin 128 → ℝ
  bias : Fin 128 → ℝ

def forwardInputs (B : Nat) : Interface.ForwardValues ℝ B ≃ ForwardInputs B where
  toFun v := ⟨Interface.forwardX v,Interface.forwardW v,Interface.forwardB v⟩
  invFun v := Fin.cases (fun j => v.x j.1) (Fin.cases (fun j => v.weight j.1)
    (Fin.cases (fun j => v.bias j.1) (fun i => i.elim0)))
  left_inv := by intro v; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro v; cases v; rfl

def forwardOutputs (B : Nat) : ChannelValues ℝ (Interface.forwardOutputs B) ≃ ForwardOutputs B where
  toFun v := ⟨fun j => v (0 : Fin 3) (j,()),v (1 : Fin 3) (),v (2 : Fin 3) ()⟩
  invFun v := Interface.forwardResult v.values v.inverse v.mean
  left_inv := by intro v; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro v; cases v; rfl

def backwardInputs (B : Nat) : Interface.BackwardValues ℝ B ≃ BackwardInputs B where
  toFun v := ⟨Interface.backwardX v,Interface.backwardDY v,Interface.backwardW v,
    Interface.backwardMean v,Interface.backwardInverse v⟩
  invFun v := Fin.cases (fun j => v.x j.1) (Fin.cases (fun j => v.upstream j.1)
    (Fin.cases (fun j => v.weight j.1) (Fin.cases (fun _ => v.mean)
      (Fin.cases (fun _ => v.inverse) (fun i => i.elim0)))))
  left_inv := by intro v; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro v; cases v; rfl

def backwardOutputs (B : Nat) : ChannelValues ℝ (Interface.backwardOutputs B) ≃ (Fin B → ℝ) where
  toFun v j := v (0 : Fin 1) (j,())
  invFun := Interface.backwardResult
  left_inv := by intro v; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro v; rfl

def affineInputs (rows : Nat) : Interface.AffineValues ℝ rows ≃ AffineInputs rows where
  toFun v := ⟨Interface.affineX v,Interface.affineDY v,Interface.affineMean v,Interface.affineInverse v⟩
  invFun v := Fin.cases (fun j => v.x j.1 j.2.1) (Fin.cases (fun j => v.upstream j.1 j.2.1)
    (Fin.cases (fun j => v.mean j.1) (Fin.cases (fun j => v.inverse j.1) (fun i => i.elim0))))
  left_inv := by intro v; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro v; cases v; rfl

def affineOutputs : ChannelValues ℝ Interface.affineOutputs ≃ AffineOutputs where
  toFun v := ⟨fun j => v (0 : Fin 2) (j,()),fun j => v (1 : Fin 2) (j,())⟩
  invFun v := Interface.affineResult v.weight v.bias
  left_inv := by intro v; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro v; cases v; rfl

def forwardView (M : Algebra ℝ) (N B pid : Nat) (eps : ℝ) (bias : Bool) :
    LogicalProgramIO ℝ ℝ (ForwardInputs B) (ForwardOutputs B) where
  core := Interface.forwardIO M N B pid eps bias
  inputs := forwardInputs B
  outputs := forwardOutputs B

def backwardView (M : Algebra ℝ) (N B pid : Nat) :
    LogicalProgramIO ℝ ℝ (BackwardInputs B) (Fin B → ℝ) where
  core := Interface.backwardIO M N B pid
  inputs := backwardInputs B
  outputs := backwardOutputs B

def affineView (M : Algebra ℝ) (rows N wc bc pid : Nat) (bias : Bool) :
    LogicalProgramIO ℝ ℝ (AffineInputs rows) AffineOutputs where
  core := Interface.affineIO M rows N wc bc pid bias
  inputs := affineInputs rows
  outputs := affineOutputs

/-- Total extension of a logical row, with no storage addresses. -/
def entry {n : Nat} (x : Fin n → ℝ) (i : Nat) : ℝ := Interface.extend x i
end
end VeriTile.Bench.Repairs.FastLayernorm.LogicalIO
