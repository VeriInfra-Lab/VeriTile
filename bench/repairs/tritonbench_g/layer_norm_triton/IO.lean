import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceModel
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ReadProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Tactic.FinCases

/-! Public finite typed channels for the three genuine source stages.
FP32 storage and arithmetic are retained by the canonical source AST; the
Structural interpreter uses its existing .real cell carrier. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Interface
open VeriTile.Triton FP.Structural SourceModel
noncomputable section

def job {α : Type} (m : Model α) (source : ComputeKernel) (pid : Nat) : Invocation α where
  kernel := source
  backend := engine m
  pids := fun ax => if ax=0 then pid else 0

namespace Forward

def inputs (N : Nat) : List IOChannel :=
  [⟨.real,[N],"X"⟩,⟨.real,[N],"W"⟩,⟨.real,[N],"Bias"⟩]
def outputs (N : Nat) : List IOChannel :=
  [⟨.real,[N],"Y"⟩,⟨.real,[],"Mean"⟩,⟨.real,[],"Rstd"⟩]
abbrev Inputs (α : Type) (N : Nat) := ChannelValues α (inputs N)
abbrev Outputs (α : Type) (N : Nat) := ChannelValues α (outputs N)

def io {α : Type} (m : Model α) (stride N B row : Nat) (eps : ℝ)
    (observe : RegionName → Nat → α → α) : ProgramIO α α where
  inputs := inputs N
  outputs := outputs N
  scratch := []
  buffers := ["X","W","Bias","Y","Mean","Rstd"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => row*stride+lane.1.val
    | ⟨1,_⟩ | ⟨2,_⟩ => fun lane => lane.1.val
    | ⟨n+3,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => row*stride+lane.1.val
    | ⟨1,_⟩ | ⟨2,_⟩ => fun _ => row
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [job m (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) row]
  encode := id
  observe := fun port => match port with
    | ⟨0,_⟩ => fun lane => observe "Y" (row*stride+lane.1.val)
    | ⟨1,_⟩ => fun _ => observe "Mean" row
    | ⟨2,_⟩ => fun _ => observe "Rstd" row
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)

def value {α : Type} [Inhabited α] {N : Nat} (vals : Inputs α N) (port : Fin 3) (j : Nat) : α :=
  match port with
  | ⟨0,_⟩ => if hj : j<N then vals (0 : Fin 3) (⟨j,hj⟩,()) else default
  | ⟨1,_⟩ => if hj : j<N then vals (1 : Fin 3) (⟨j,hj⟩,()) else default
  | ⟨2,_⟩ => if hj : j<N then vals (2 : Fin 3) (⟨j,hj⟩,()) else default
  | ⟨n+3,h⟩ => False.elim (by omega)

def result {α : Type} {N : Nat} (ys : Fin N → α) (mean inverse : α) : Outputs α N
  | ⟨0,_⟩ => fun lane => ys lane.1
  | ⟨1,_⟩ => fun _ => mean
  | ⟨2,_⟩ => fun _ => inverse
  | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)
end Forward

namespace Final

def inputs (M BN : Nat) : List IOChannel :=
  [⟨.real,[M,BN],"DW"⟩,⟨.real,[M,BN],"DB"⟩]
def outputs (BN : Nat) : List IOChannel :=
  [⟨.real,[BN],"FINAL_DW"⟩,⟨.real,[BN],"FINAL_DB"⟩]
abbrev Inputs (α : Type) (M BN : Nat) := ChannelValues α (inputs M BN)
abbrev Outputs (α : Type) (BN : Nat) := ChannelValues α (outputs BN)

def io {α : Type} (m : Model α) (M N BM BN pid : Nat)
    (observe : RegionName → Nat → α → α) : ProgramIO α α where
  inputs := inputs M BN
  outputs := outputs BN
  scratch := []
  buffers := ["DW","DB","FINAL_DW","FINAL_DB"]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun lane => lane.1.val*N+(pid*BN+lane.2.1.val)
    | ⟨n+2,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun lane => pid*BN+lane.2.1.val<N
    | ⟨n+2,h⟩ => False.elim (by simp [inputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun lane => pid*BN+lane.1.val
    | ⟨n+2,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun lane => pid*BN+lane.1.val<N
    | ⟨n+2,h⟩ => False.elim (by simp [outputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [job m (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) pid]
  encode := id
  observe := fun port => match port with
    | ⟨0,_⟩ => fun lane => observe "FINAL_DW" (pid*BN+lane.1.val)
    | ⟨1,_⟩ => fun lane => observe "FINAL_DB" (pid*BN+lane.1.val)
    | ⟨n+2,h⟩ => False.elim (by simp [outputs] at h)

def value {α : Type} [Inhabited α] {M BN : Nat} (vals : Inputs α M BN)
    (port : Fin 2) (row : Nat) (j : Fin BN) : α :=
  match port with
  | ⟨0,_⟩ => if hr : row<M then vals (0 : Fin 2) (⟨row,hr⟩,j,()) else default
  | ⟨1,_⟩ => if hr : row<M then vals (1 : Fin 2) (⟨row,hr⟩,j,()) else default
  | ⟨n+2,h⟩ => False.elim (by omega)

def result {α : Type} {BN : Nat} (dw db : Fin BN → α) : Outputs α BN
  | ⟨0,_⟩ => fun lane => dw lane.1
  | ⟨1,_⟩ => fun lane => db lane.1
  | ⟨n+2,h⟩ => False.elim (by simp [outputs] at h)
end Final
end
end VeriTile.Bench.Repairs.LayerNormTriton.Interface

namespace VeriTile.Bench.Repairs.LayerNormTriton

/-- Exact increasing source while-counter values. For G>0 this reaches M in
the least number of increments; the geometry module supplies coverage/bounds. -/
def groupRows (M G group : Nat) : List Nat :=
  List.range' group ((M-group+G-1)/G) G

end VeriTile.Bench.Repairs.LayerNormTriton

/-! Grouped backward retains the original declared DX window and frame.
The full DX result is observed, while the actual source updates only visited
lanes. The original broad M×N writable window is retained; this interface
does not strengthen the previous frame. Zero stride, zero loop step and
unvisited rows preserve their existing read-value behavior. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Interface.Grouped
open VeriTile.Triton FP.Structural SourceModel
noncomputable section

def inputs (M N B : Nat) : List IOChannel :=
  [⟨.real,[M,B],"X"⟩,⟨.real,[M,B],"DY"⟩,⟨.real,[B],"W"⟩,
   ⟨.real,[M],"Mean"⟩,⟨.real,[M],"Rstd"⟩,⟨.real,[M,N],"DX"⟩]
def outputs (M N B : Nat) : List IOChannel :=
  [⟨.real,[M,N],"DX"⟩,⟨.real,[B],"DW"⟩,⟨.real,[B],"DB"⟩]
abbrev Inputs (α : Type) (M N B : Nat) := ChannelValues α (inputs M N B)
abbrev Outputs (α : Type) (M N B : Nat) := ChannelValues α (outputs M N B)

def layout {α : Type} (m : Model α) (stride M N G B group : Nat)
    (observe : RegionName → Nat → α → α) : ProgramIO α α where
  inputs := inputs M N B
  outputs := outputs M N B
  scratch := []
  buffers := ["X","DY","W","Mean","Rstd","DX","DW","DB"]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun lane => lane.1.val*stride+lane.2.1.val
    | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => fun lane => lane.1.val
    | ⟨5,_⟩ => fun lane => lane.1.val*stride+lane.2.1.val
    | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun lane => lane.2.1.val<N
    | ⟨2,_⟩ => fun lane => lane.1.val<N
    | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ => fun _ => True
    | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.1.val*stride+lane.2.1.val
    | ⟨1,_⟩ | ⟨2,_⟩ => fun lane => group*N+lane.1.val
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ => fun _ => True
    | ⟨1,_⟩ | ⟨2,_⟩ => fun lane => lane.1.val<N
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [job m
    (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) group]
  encode := id
  observe := fun port => match port with
    | ⟨0,_⟩ => fun lane => observe "DX" (lane.1.val*stride+lane.2.1.val)
    | ⟨1,_⟩ => fun lane => observe "DW" (group*N+lane.1.val)
    | ⟨2,_⟩ => fun lane => observe "DB" (group*N+lane.1.val)
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)

def io {α : Type} (m : Model α) (stride M N G B group : Nat)
    (observe : RegionName → Nat → α → α) : ReadProgramIO α α where
  toProgramIO := layout m stride M N G B group observe
  resultRead := (layout m stride M N G B group observe).write
  resultMask := (layout m stride M N G B group observe).outputMask

def x {α : Type} [Inhabited α] {M N B : Nat} (vals : Inputs α M N B) (row : Nat) (j : Fin B) : α :=
  if hr : row<M then vals (0 : Fin 6) (⟨row,hr⟩,j,()) else default

def dy {α : Type} [Inhabited α] {M N B : Nat} (vals : Inputs α M N B) (row : Nat) (j : Fin B) : α :=
  if hr : row<M then vals (1 : Fin 6) (⟨row,hr⟩,j,()) else default

def w {α : Type} {M N B : Nat} (vals : Inputs α M N B) (j : Fin B) : α :=
  vals (2 : Fin 6) (j,())
def mean {α : Type} [Inhabited α] {M N B : Nat} (vals : Inputs α M N B) (row : Nat) : α :=
  if hr : row<M then vals (3 : Fin 6) (⟨row,hr⟩,()) else default

def inverse {α : Type} [Inhabited α] {M N B : Nat} (vals : Inputs α M N B) (row : Nat) : α :=
  if hr : row<M then vals (4 : Fin 6) (⟨row,hr⟩,()) else default

def priorDX {α : Type} {M N B : Nat} (vals : Inputs α M N B) (row : Fin M) (j : Fin N) : α :=
  vals (5 : Fin 6) (row,j,())

def result {α : Type} {M N B : Nat} (dx : Fin M → Fin N → α) (dw db : Fin B → α) : Outputs α M N B
  | ⟨0,_⟩ => fun lane => dx lane.1 lane.2.1
  | ⟨1,_⟩ => fun lane => dw lane.1
  | ⟨2,_⟩ => fun lane => db lane.1
  | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)
end
end VeriTile.Bench.Repairs.LayerNormTriton.Interface.Grouped

namespace VeriTile.Bench.Repairs.LayerNormTriton.GroupedRealIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section

def io (m : Model ℝ) (stride M N G B group : Nat) : ReadProgramIO ℝ ℝ :=
  { Grouped.io m stride M N G B group (fun _ _ => id) with
    resultMask := fun port _ => match port with
      | ⟨0,_⟩ => fun lane => lane.1.val∈groupRows M G group
      | ⟨1,_⟩ | ⟨2,_⟩ => fun lane => lane.1.val<N
      | ⟨n+3,h⟩ => False.elim (by simp [Grouped.io,Grouped.layout,Grouped.outputs] at h) }

end
end VeriTile.Bench.Repairs.LayerNormTriton.GroupedRealIO

namespace VeriTile.Bench.Repairs.LayerNormTriton.Logical
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section

/-- Complete finite row inputs, before any source computation. -/
structure ForwardInputs (N : Nat) where
  x : Fin N → ℝ
  weight : Fin N → ℝ
  bias : Fin N → ℝ

/-- All three genuine forward outputs, including both saved statistics. -/
structure ForwardOutputs (N : Nat) where
  y : Fin N → ℝ
  mean : ℝ
  inverse : ℝ

def forwardInputs (N : Nat) : Forward.Inputs ℝ N ≃ ForwardInputs N where
  toFun := fun raw => ⟨fun i => raw (0 : Fin 3) (i,()),
    fun i => raw (1 : Fin 3) (i,()),fun i => raw (2 : Fin 3) (i,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun i => xs.x i.1
    | ⟨1,_⟩ => fun i => xs.weight i.1
    | ⟨2,_⟩ => fun i => xs.bias i.1
    | ⟨n+3,h⟩ => False.elim (by simp [Forward.inputs] at h)
  left_inv := by
    intro raw
    funext port
    fin_cases port <;> funext i <;> rcases i with ⟨i,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def forwardOutputs (N : Nat) : Forward.Outputs ℝ N ≃ ForwardOutputs N where
  toFun := fun raw => ⟨fun i => raw (0 : Fin 3) (i,()),raw (1 : Fin 3) (),raw (2 : Fin 3) ()⟩
  invFun := fun ys => Forward.result ys.y ys.mean ys.inverse
  left_inv := by
    intro raw
    funext port
    fin_cases port <;> funext i
    · rcases i with ⟨i,⟨⟩⟩; rfl
    · cases i; rfl
    · cases i; rfl
  right_inv := by intro ys; cases ys; rfl

/-- Total logical-column lookup; all finite row entries are still retained. -/
def extend {N : Nat} (xs : Fin N → ℝ) (j : Nat) : ℝ :=
  if hj : j<N then xs ⟨j,hj⟩ else 0

def ForwardInputs.xAt {N : Nat} (xs : ForwardInputs N) := extend xs.x
def ForwardInputs.weightAt {N : Nat} (xs : ForwardInputs N) := extend xs.weight
def ForwardInputs.biasAt {N : Nat} (xs : ForwardInputs N) := extend xs.bias

def forwardIO (m : Model ℝ) (stride N B row : Nat) (eps : ℝ) :
    LogicalProgramIO ℝ ℝ (ForwardInputs N) (ForwardOutputs N) where
  core := Forward.io m stride N B row eps (fun _ _ => id)
  inputs := forwardInputs N
  outputs := forwardOutputs N

/-- Complete padded matrices, saved row statistics and original DX snapshot.
Inactive B columns and every prior DX coordinate remain represented. -/
structure GroupedInputs (M N B : Nat) where
  x : Fin M → Fin B → ℝ
  dy : Fin M → Fin B → ℝ
  weight : Fin B → ℝ
  mean : Fin M → ℝ
  inverse : Fin M → ℝ
  priorDX : Fin M → Fin N → ℝ

structure GroupedOutputs (M N B : Nat) where
  dx : Fin M → Fin N → ℝ
  dw : Fin B → ℝ
  db : Fin B → ℝ

def groupedInputs (M N B : Nat) : Grouped.Inputs ℝ M N B ≃ GroupedInputs M N B where
  toFun := fun raw => ⟨fun r j => raw (0 : Fin 6) (r,j,()),
    fun r j => raw (1 : Fin 6) (r,j,()),fun j => raw (2 : Fin 6) (j,()),
    fun r => raw (3 : Fin 6) (r,()),fun r => raw (4 : Fin 6) (r,()),
    fun r j => raw (5 : Fin 6) (r,j,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun i => xs.x i.1 i.2.1
    | ⟨1,_⟩ => fun i => xs.dy i.1 i.2.1
    | ⟨2,_⟩ => fun i => xs.weight i.1
    | ⟨3,_⟩ => fun i => xs.mean i.1
    | ⟨4,_⟩ => fun i => xs.inverse i.1
    | ⟨5,_⟩ => fun i => xs.priorDX i.1 i.2.1
    | ⟨n+6,h⟩ => False.elim (by simp [Grouped.inputs] at h)
  left_inv := by
    intro raw
    funext port
    fin_cases port <;> funext i
    all_goals first | (rcases i with ⟨r,j,⟨⟩⟩; rfl) | (rcases i with ⟨j,⟨⟩⟩; rfl)
  right_inv := by intro xs; cases xs; rfl

def groupedOutputs (M N B : Nat) : Grouped.Outputs ℝ M N B ≃ GroupedOutputs M N B where
  toFun := fun raw => ⟨fun r j => raw (0 : Fin 3) (r,j,()),
    fun j => raw (1 : Fin 3) (j,()),fun j => raw (2 : Fin 3) (j,())⟩
  invFun := fun ys => Grouped.result ys.dx ys.dw ys.db
  left_inv := by
    intro raw
    funext port
    fin_cases port <;> funext i
    all_goals first | (rcases i with ⟨r,j,⟨⟩⟩; rfl) | (rcases i with ⟨j,⟨⟩⟩; rfl)
  right_inv := by intro ys; cases ys; rfl

/-- The same total row lookups as the old reference, now on named tensors. -/
def GroupedInputs.xRow {M N B : Nat} (xs : GroupedInputs M N B) (row : Nat) : Fin B → ℝ :=
  if hr : row<M then xs.x ⟨row,hr⟩ else fun _ => 0

def GroupedInputs.dyRow {M N B : Nat} (xs : GroupedInputs M N B) (row : Nat) : Fin B → ℝ :=
  if hr : row<M then xs.dy ⟨row,hr⟩ else fun _ => 0

def GroupedInputs.meanAt {M N B : Nat} (xs : GroupedInputs M N B) (row : Nat) : ℝ :=
  if hr : row<M then xs.mean ⟨row,hr⟩ else 0

def GroupedInputs.inverseAt {M N B : Nat} (xs : GroupedInputs M N B) (row : Nat) : ℝ :=
  if hr : row<M then xs.inverse ⟨row,hr⟩ else 0

/-- The entire original ReadProgramIO judgment is fixed, including resultRead,
visited-row resultMask, broad DX writable window, default read semantics and frame. -/
def groupedIO (m : Model ℝ) (stride M N G B group : Nat) :
    VeriTile.Triton.LogicalIO (Grouped.Inputs ℝ M N B) (Grouped.Outputs ℝ M N B)
      (GroupedInputs M N B) (GroupedOutputs M N B)
      ((GroupedRealIO.io m stride M N G B group).Implements) where
  inputs := groupedInputs M N B
  outputs := groupedOutputs M N B

structure FinalInputs (M BN : Nat) where
  dw : Fin M → Fin BN → ℝ
  db : Fin M → Fin BN → ℝ

structure FinalOutputs (BN : Nat) where
  dw : Fin BN → ℝ
  db : Fin BN → ℝ

def finalInputs (M BN : Nat) : Final.Inputs ℝ M BN ≃ FinalInputs M BN where
  toFun := fun raw => ⟨fun r j => raw (0 : Fin 2) (r,j,()),fun r j => raw (1 : Fin 2) (r,j,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun i => xs.dw i.1 i.2.1
    | ⟨1,_⟩ => fun i => xs.db i.1 i.2.1
    | ⟨n+2,h⟩ => False.elim (by simp [Final.inputs] at h)
  left_inv := by
    intro raw
    funext port
    fin_cases port <;> funext i <;> rcases i with ⟨r,j,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def finalOutputs (BN : Nat) : Final.Outputs ℝ BN ≃ FinalOutputs BN where
  toFun := fun raw => ⟨fun j => raw (0 : Fin 2) (j,()),fun j => raw (1 : Fin 2) (j,())⟩
  invFun := fun ys => Final.result ys.dw ys.db
  left_inv := by
    intro raw
    funext port
    fin_cases port <;> funext i <;> rcases i with ⟨j,⟨⟩⟩ <;> rfl
  right_inv := by intro ys; cases ys; rfl

def finalIO (m : Model ℝ) (M N BN pid : Nat) :
    LogicalProgramIO ℝ ℝ (FinalInputs M BN) (FinalOutputs BN) where
  core := Final.io m M N 32 BN pid (fun _ _ => id)
  inputs := finalInputs M BN
  outputs := finalOutputs BN

end
end VeriTile.Bench.Repairs.LayerNormTriton.Logical
