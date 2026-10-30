import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.RealOutput
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Grid
import VeriTile.Triton.Float.ProgramEquivalence
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option autoImplicit false

/-- Mathematical scale only. The native observer separately binds the actual
host/JIT FP32 scale word, preserving the source scalar ABI. -/
def realScale (c : Shape) : ℝ := 1/Real.sqrt (c.D:ℝ)

theorem realScale_ne_zero (c : Shape) : realScale c≠0 := by
  exact div_ne_zero one_ne_zero (ne_of_gt (Real.sqrt_pos.mpr (Nat.cast_pos.mpr c.dim_pos)))

def inputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.mainSize],"q"⟩,⟨.real,[c.mainSize],"k"⟩,⟨.real,[c.mainSize],"v"⟩]
def outputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.mainSize],"o"⟩,⟨.real,[c.historySize],"h"⟩]

def memory (α : Type) (c : Shape) (storeFlag : Bool) : MemoryIO α where
  inputs := inputs c
  outputs := outputs c
  scratch := []
  buffers := ["q","k","v","o","h"]
  read | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i | ⟨2,_⟩,_,i=>i.1.val
  inputMask := fun _ _ _=>True
  write | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i=>i.1.val
  outputMask | ⟨0,_⟩,_,_=>True | ⟨1,_⟩,_,_=>storeFlag=Bool.true
  scratchWrite := fun i=>Fin.elim0 i
  scratchMask := fun i=>Fin.elim0 i

def invocation (α : Type) [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (head : Nat) : Invocation (Term α) where
  kernel := Program.kernel c scale storeFlag ifcond
  backend := algebra
  pids := fun ax=>if ax=0 then head else 0
  numPids := some (fun ax=>if ax=0 then c.heads else 1)

def schedule (α : Type) [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) : List (Invocation (Term α)) :=
  if SharedGrid.Empty c then [] else (List.range c.heads).map (invocation α c scale storeFlag ifcond)

/-- Actual shared quote with FP32 load/computation/cast sites, every head and
every chunk in source order, O and optional H. `.real` is the existing symbolic
cell carrier; Physical realizes these outputs as FP32 cells at ABI addresses. -/
def io {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) : ProgramIO α (Term α) where
  toMemoryIO := memory α c storeFlag
  schedule := fun _=>schedule α c scale storeFlag ifcond
  encode := Term.input
  observe := fun _ _ t=>t.evaluate ops literal

abbrev Values (α : Type) (c : Shape) := ChannelValues α (inputs c)

def array {α : Type} [Inhabited α] (c : Shape) (values : Values α c) (which : Fin 3) (off : Nat) : α :=
  if h : off<c.mainSize then
    match which with
    | ⟨0,_⟩=>values (0 : Fin 3) (⟨off,h⟩,PUnit.unit)
    | ⟨1,_⟩=>values (1 : Fin 3) (⟨off,h⟩,PUnit.unit)
    | ⟨2,_⟩=>values (2 : Fin 3) (⟨off,h⟩,PUnit.unit)
  else default

def logicalInputs {α : Type} [Inhabited α] (c : Shape) (values : Values α c) : Inputs α where
  q := fun h r d=>array c values 0 (c.mainOffset h r d)
  k := fun h r d=>array c values 1 (c.mainOffset h r d)
  v := fun h r d=>array c values 2 (c.mainOffset h r d)

def mainHead (c : Shape) (off : Nat) := off/c.headSize
def mainRow (c : Shape) (off : Nat) := off%c.headSize/c.D
def mainColumn (c : Shape) (off : Nat) := off%c.D

def historyHead (c : Shape) (off : Nat) := off/c.historyHead
def historyChunk (c : Shape) (off : Nat) := off%c.historyHead/(c.D*c.D)
def historyRow (c : Shape) (off : Nat) := off%c.historyHead/c.D%c.D
def historyColumn (c : Shape) (off : Nat) := off%c.D

/-- Independent ordered primitive recurrence, without states or executions. -/
def reference {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (ifcond : Bool) (values : Values α c) : ChannelValues α (outputs c)
  | ⟨0,_⟩,i=>ops.cast (Reference.output ops c (logicalInputs c values) (literal scale)
      (mainHead c i.1.val) (mainRow c i.1.val/32) ifcond (mainRow c i.1.val%32) (mainColumn c i.1.val))
  | ⟨1,_⟩,i=>ops.cast (Reference.carry ops c (logicalInputs c values) (literal scale)
      (historyHead c i.1.val) ifcond (historyChunk c i.1.val) (historyRow c i.1.val) (historyColumn c i.1.val))

#specNonCircular reference avoiding [FP.Structural.exec,FP.Structural.run,SharedGrid.result,SharedLoop.state]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Pair
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

/-- Candidate IO selects the existing shifted-index source; connection proofs remain in Proofs.Pair. -/
def kernel (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) : ComputeKernel :=
  shiftedSource "q" "k" "v" "h" "o" c.headSize c.D 1 c.historyHead c.D c.T scale
    c.D c.chunks storeFlag ifcond

def invocation (α : Type) [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (head : Nat) : Invocation (Term α) :=
  { KernelIO.invocation α c scale storeFlag ifcond head with kernel := kernel c scale storeFlag ifcond }

def schedule (α : Type) [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) : List (Invocation (Term α)) :=
  if SharedGrid.Empty c then [] else (List.range c.heads).map (invocation α c scale storeFlag ifcond)

def variant {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    ProgramIO.Variant (KernelIO.io ops literal c scale storeFlag ifcond) where
  schedule := fun _ => schedule α c scale storeFlag ifcond
  observe := fun _ _ t => t.evaluate ops literal

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Pair

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.RealIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

/-- Complete actual source schedule and the declared Q/K/V, O and optional H
windows. The shared IO retains the exact source FP32 precision sites. -/
def io (c : Shape) (storeFlag ifcond : Bool) : ProgramIO ℝ (Term ℝ) :=
  KernelIO.io realOps id c (KernelIO.realScale c) storeFlag ifcond

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.RealIO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.LogicalIO
open VeriTile Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

def mainCoordinates (c : Shape) : Fin c.mainSize ≃ (Fin c.heads × Fin c.T × Fin c.D) :=
  (finProdFinEquiv : (Fin c.heads × Fin (c.T*c.D)) ≃ Fin c.mainSize).symm.trans
    (Equiv.prodCongr (Equiv.refl _) finProdFinEquiv.symm)

def historyCoordinates (c : Shape) :
    Fin c.historySize ≃ (Fin c.heads × Fin c.chunks × Fin c.D × Fin c.D) :=
  (finCongr (by simp [Shape.historySize,Shape.historyHead,Nat.mul_assoc])).trans
    ((finProdFinEquiv : (Fin c.heads × Fin (c.chunks*(c.D*c.D))) ≃
        Fin (c.heads*(c.chunks*(c.D*c.D)))).symm.trans
      (Equiv.prodCongr (Equiv.refl _)
        (finProdFinEquiv.symm.trans (Equiv.prodCongr (Equiv.refl _) finProdFinEquiv.symm))))

structure Inputs (c : Shape) where
  arrays : AttentionFwdTriton1.Whole001.Inputs ℝ
  backing : KernelIO.Values ℝ c
  connection : arrays = KernelIO.logicalInputs c backing

def Inputs.query (xs : Inputs c) := xs.arrays.q
def Inputs.key (xs : Inputs c) := xs.arrays.k
def Inputs.value (xs : Inputs c) := xs.arrays.v

structure Outputs (c : Shape) where
  output : Fin c.heads → Fin c.T → Fin c.D → ℝ
  history : Fin c.heads → Fin c.chunks → Fin c.D → Fin c.D → ℝ

def inputsEquiv (c : Shape) : KernelIO.Values ℝ c ≃ Inputs c where
  toFun := fun raw => ⟨KernelIO.logicalInputs c raw,raw,rfl⟩
  invFun := Inputs.backing
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,backing,connection⟩
    cases connection
    rfl

def outputsEquiv (c : Shape) : ChannelValues ℝ (KernelIO.outputs c) ≃ Outputs c where
  toFun := fun values => {
    output := fun h row col => values (0 : Fin 2) ((mainCoordinates c).symm (h,row,col),())
    history := fun h chunk row col => values (1 : Fin 2) ((historyCoordinates c).symm (h,chunk,row,col),()) }
  invFun := fun values => Fin.cases (fun j =>
    let point := mainCoordinates c j.1
    values.output point.1 point.2.1 point.2.2)
    (Fin.cases (fun j =>
      let point := historyCoordinates c j.1
      values.history point.1 point.2.1 point.2.2.1 point.2.2.2) (fun i => Fin.elim0 i))
  left_inv := by
    intro values
    funext o j
    fin_cases o
    · rcases j with ⟨j,⟨⟩⟩
      change values (0 : Fin 2) ((mainCoordinates c).symm (mainCoordinates c j),()) = values (0 : Fin 2) (j,())
      rw [Equiv.symm_apply_apply]
    · rcases j with ⟨j,⟨⟩⟩
      change values (1 : Fin 2) ((historyCoordinates c).symm (historyCoordinates c j),()) = values (1 : Fin 2) (j,())
      rw [Equiv.symm_apply_apply]
  right_inv := by
    rintro ⟨output,history⟩
    apply congrArg₂ Outputs.mk
    · funext h row col
      change (let point := mainCoordinates c ((mainCoordinates c).symm (h,row,col));
        output point.1 point.2.1 point.2.2) = output h row col
      rw [Equiv.apply_symm_apply]
    · funext h chunk row col
      change (let point := historyCoordinates c ((historyCoordinates c).symm (h,chunk,row,col));
        history point.1 point.2.1 point.2.2.1 point.2.2.2) = history h chunk row col
      rw [Equiv.apply_symm_apply]

def io (c : Shape) (storeFlag ifcond : Bool) :
    LogicalProgramIO ℝ (Term ℝ) (Inputs c) (Outputs c) where
  core := RealIO.io c storeFlag ifcond
  inputs := inputsEquiv c
  outputs := outputsEquiv c

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.LogicalIO
