import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeReal
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.fused_layernorm_triton.Contract
import bench.tritonbench_g.fused_layernorm_triton.Proofs.SourcePieces
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_FPWelfordTree
import VeriTile.Triton.Float.ProgramIO

/-! Complete shared row-grid IO data. Real and native interpretations select
only the backend and observer; the hoisted source changes only the schedule. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (row : Nat) : Invocation α where
  kernel := Contract.source X W Bias Y Mean Rstd rows D c
  backend := withEquality M eq32
  pids := fun dim => if dim=0 then row else 0

abbrev inputs (X W Bias : RegionName) (rows D : Nat) : List IOChannel :=
  [⟨.real,[rows,D],X⟩,⟨.real,[D],W⟩,⟨.real,[D],Bias⟩]

abbrev outputs (Y Mean Rstd : RegionName) (rows D : Nat) : List IOChannel :=
  [⟨.real,[rows],Mean⟩,⟨.real,[rows],Rstd⟩,⟨.bf16,[rows,D],Y⟩]

abbrev memory (α : Type) (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) : MemoryIO α where
  inputs := inputs X W Bias rows D
  outputs := outputs Y Mean Rstd rows D
  scratch := []
  buffers := [X,W,Bias,Y,Mean,Rstd]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.2.1.val+D*lane.1.val
    | ⟨1,_⟩ => fun lane => lane.1.val
    | ⟨2,_⟩ => fun lane => lane.1.val
    | ⟨n+3,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.1.val
    | ⟨1,_⟩ => fun lane => lane.1.val
    | ⟨2,_⟩ => fun lane => lane.2.1.val+D*lane.1.val
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0

def io (M : Algebra α) (eq32 : α → α → Bool) (observe : α → α)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration) : ProgramIO α α where
  toMemoryIO := memory α X W Bias Y Mean Rstd rows D
  schedule := fun _ => (List.range rows).map (invocation M eq32 X W Bias Y Mean Rstd rows D c)
  encode := id
  observe := fun _ _ => observe

abbrev arrayX (X W Bias : RegionName) (rows D : Nat)
    (vals : ChannelValues α (inputs X W Bias rows D)) (r col : Nat) : α :=
  if hr : r<rows then if hc : col<D then
    vals ⟨0,by simp [inputs]⟩ (⟨r,hr⟩,⟨col,hc⟩,()) else default else default

abbrev arrayW (X W Bias : RegionName) (rows D : Nat)
    (vals : ChannelValues α (inputs X W Bias rows D)) (col : Nat) : α :=
  if hc : col<D then vals ⟨1,by simp [inputs]⟩ (⟨col,hc⟩,()) else default

abbrev arrayB (X W Bias : RegionName) (rows D : Nat)
    (vals : ChannelValues α (inputs X W Bias rows D)) (col : Nat) : α :=
  if hc : col<D then vals ⟨2,by simp [inputs]⟩ (⟨col,hc⟩,()) else default

abbrev expected (X W Bias Y Mean Rstd : RegionName) (rows D : Nat)
    (f : (Nat → Nat → α) → (Nat → α) → (Nat → α) → Nat → α×α×(Nat → α))
    (vals : ChannelValues α (inputs X W Bias rows D)) : ChannelValues α (outputs Y Mean Rstd rows D) :=
  let result := f (arrayX X W Bias rows D vals) (arrayW X W Bias rows D vals) (arrayB X W Bias rows D vals)
  fun port => match port with
    | ⟨0,_⟩ => fun lane => (result lane.1.val).1
    | ⟨1,_⟩ => fun lane => (result lane.1.val).2.1
    | ⟨2,_⟩ => fun lane => (result lane.1.val).2.2 lane.2.1.val
    | ⟨n+3,h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (row : Nat) : Invocation α where
  kernel := SourcePieces.source hoist X W Bias Y Mean Rstd rows D c.block c.tree
  backend := withEquality M eq32
  pids := fun dim => if dim=0 then row else 0

abbrev inputs := PublicIO.inputs

abbrev outputs := PublicIO.outputs

def io (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool) (observe : α → α)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration) : ProgramIO α α where
  toMemoryIO := PublicIO.memory α X W Bias Y Mean Rstd rows D
  schedule := fun _ => (List.range rows).map (invocation hoist M eq32 X W Bias Y Mean Rstd rows D c)
  encode := id
  observe := fun _ _ => observe

abbrev arrayX := @PublicIO.arrayX

abbrev arrayW := @PublicIO.arrayW

abbrev arrayB := @PublicIO.arrayB

abbrev expected := @PublicIO.expected

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.PairIO

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
structure TensorInputs (α : Type) (rows D : Nat) where
  x : Fin (rows) → Fin (D) → α
  weight : Fin (D) → α
  bias : Fin (D) → α

def inputCoordinates (α : Type) (X W Bias : RegionName) (rows D : Nat) : ChannelValues α (inputs X W Bias rows D) ≃ TensorInputs α rows D where
  toFun := fun xs => ⟨Values.curryEquiv α .real [rows, D] (xs (0 : Fin 3)),
    Values.curryEquiv α .real [D] (xs (1 : Fin 3)),
    Values.curryEquiv α .real [D] (xs (2 : Fin 3))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [rows, D]).symm xs.x) (Fin.cases ((Values.curryEquiv α .real [D]).symm xs.weight) (Fin.cases ((Values.curryEquiv α .real [D]).symm xs.bias) (fun i => Fin.elim0 i))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [rows, D]).symm_apply_apply (xs (0 : Fin 3))
    · exact (Values.curryEquiv α .real [D]).symm_apply_apply (xs (1 : Fin 3))
    · exact (Values.curryEquiv α .real [D]).symm_apply_apply (xs (2 : Fin 3))
  right_inv := by intro xs; cases xs; rfl

structure TensorOutputs (α : Type) (rows D : Nat) where
  mean : Fin (rows) → α
  inverse : Fin (rows) → α
  y : Fin (rows) → Fin (D) → α

def outputCoordinates (α : Type) (Y Mean Rstd : RegionName) (rows D : Nat) : ChannelValues α (outputs Y Mean Rstd rows D) ≃ TensorOutputs α rows D where
  toFun := fun xs => ⟨Values.curryEquiv α .real [rows] (xs (0 : Fin 3)),
    Values.curryEquiv α .real [rows] (xs (1 : Fin 3)),
    Values.curryEquiv α .bf16 [rows, D] (xs (2 : Fin 3))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [rows]).symm xs.mean) (Fin.cases ((Values.curryEquiv α .real [rows]).symm xs.inverse) (Fin.cases ((Values.curryEquiv α .bf16 [rows, D]).symm xs.y) (fun i => Fin.elim0 i))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [rows]).symm_apply_apply (xs (0 : Fin 3))
    · exact (Values.curryEquiv α .real [rows]).symm_apply_apply (xs (1 : Fin 3))
    · exact (Values.curryEquiv α .bf16 [rows, D]).symm_apply_apply (xs (2 : Fin 3))
  right_inv := by intro xs; cases xs; rfl

def tensorX (xs : TensorInputs ℝ rows D) (r col : Nat) : ℝ :=
  if hr : r<rows then if hc : col<D then xs.x ⟨r,hr⟩ ⟨col,hc⟩ else 0 else 0

def tensorW (xs : TensorInputs ℝ rows D) (col : Nat) : ℝ :=
  if hc : col<D then xs.weight ⟨col,hc⟩ else 0

def tensorB (xs : TensorInputs ℝ rows D) (col : Nat) : ℝ :=
  if hc : col<D then xs.bias ⟨col,hc⟩ else 0

/-- A mathematical tuple of mean/rstd/Y in genuine logical tensor coordinates. -/
def logicalExpected (rows D : Nat)
    (f : (Nat → Nat → ℝ) → (Nat → ℝ) → (Nat → ℝ) → Nat → ℝ×ℝ×(Nat → ℝ))
    (xs : TensorInputs ℝ rows D) : TensorOutputs ℝ rows D :=
  let value := f (tensorX xs) (tensorW xs) (tensorB xs)
  { mean := fun row => (value row.val).1
    inverse := fun row => (value row.val).2.1
    y := fun row col => (value row.val).2.2 col.val }

def logicalIO (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName)
    (rows D : Nat) (c : Contract.Configuration) :
    LogicalProgramIO ℝ ℝ (TensorInputs ℝ rows D) (TensorOutputs ℝ rows D) where
  core := io (ideal rest) realEqual id X W Bias Y Mean Rstd rows D c
  inputs := inputCoordinates ℝ X W Bias rows D
  outputs := outputCoordinates ℝ Y Mean Rstd rows D
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO
