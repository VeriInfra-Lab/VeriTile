import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Tile
import bench.tritonbench_g.kldiv_triton.Proofs.Math
import VeriTile.Triton.Float.ProgramEquivalence

/-! Public channel and launch data shared by the Real and native judgments.
The complete row schedule runs the selected sources at their original FP32
precision. Reduced forward output contains one cell per source chunk. -/
namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.IO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
noncomputable section

def forwardInputs (rows N : Nat) : List IOChannel :=
  [⟨.real,[rows,N],"y_ptr"⟩,⟨.real,[rows,N],"gt_ptr"⟩]
def backwardInputs (rows N : Nat) : List IOChannel :=
  [⟨.real,[rows,N],"target_ptr"⟩]

/-- Logical row-stride geometry, independent of memory and execution. -/
def rowAddress {rows columns : Nat} (stride : Nat) (q : Fin rows × Fin columns) : Nat :=
  q.1.val*stride+q.2.val

def launch {α : Type} (m : Tile.Model α) (source : ComputeKernel) (pid : Nat) :
    FP.Structural.Invocation α where
  kernel := source
  backend := m.alg
  pids := fun ax => if ax=0 then pid else 0

def forward {α : Type} (m : Tile.Model α) (rows sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) (work : List Nat) : ProgramIO α α where
  inputs := forwardInputs rows N
  outputs := [⟨.real,[rows,outputCount N B reduction],"loss_ptr"⟩]
  scratch := []
  buffers := ["y_ptr","gt_ptr","loss_ptr"]
  read := fun i _ => Fin.cases
    (fun j => j.1.val*sy+j.2.1.val)
    (fun k => Fin.cases (fun j => j.1.val*st+j.2.1.val) (fun impossible => Fin.elim0 impossible) k) i
  inputMask := fun _ _ _ => True
  write := fun i _ => Fin.cases
    (fun j => j.1.val*so+j.2.1.val) (fun impossible => Fin.elim0 impossible) i
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => work.map (launch m (forwardSource sy st so N B flag reduction))
  encode := id
  observe := fun _ _ => id

def backward {α : Type} (m : Tile.Model α) (rows so st N B : Nat)
    (flag : Bool) (work : List Nat) : ProgramIO α α where
  inputs := backwardInputs rows N
  outputs := [⟨.real,[rows,N],"input_ptr"⟩]
  scratch := []
  buffers := ["target_ptr","input_ptr"]
  read := fun i _ => Fin.cases
    (fun j => j.1.val*st+j.2.1.val) (fun impossible => Fin.elim0 impossible) i
  inputMask := fun _ _ _ => True
  write := fun i _ => Fin.cases
    (fun j => j.1.val*so+j.2.1.val) (fun impossible => Fin.elim0 impossible) i
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => work.map (launch m (backwardSource so st N B flag))
  encode := id
  observe := fun _ _ => id

def forwardCandidate {α : Type} (m : Tile.Model α) (rows sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) (work : List Nat) :
    ProgramIO.Variant (forward m rows sy st so N B flag reduction work) where
  schedule := fun _ => work.map (launch m (forwardCandidateSource sy st so N B flag reduction))
  observe := fun _ _ => id

def backwardCandidate {α : Type} (m : Tile.Model α) (rows so st N B : Nat)
    (flag : Bool) (work : List Nat) : ProgramIO.Variant (backward m rows so st N B flag work) where
  schedule := fun _ => work.map (launch m (backwardCandidateSource so st N B flag))
  observe := fun _ _ => id

/-- A logical row, with an irrelevant extension outside the declared channel. -/
def rowsOf {α : Type} [Inhabited α] {rows N : Nat}
    (xs : Values α .real [rows,N]) (r c : Nat) : α :=
  if hr : r<rows then if hc : c<N then xs (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) else default else default

def forwardFormula {α : Type} [Inhabited α] {rows N columns : Nat}
    (f : (Nat → α) → (Nat → α) → Nat → α)
    (vals : ChannelValues α (forwardInputs rows N)) :
    ChannelValues α [⟨.real,[rows,columns],"loss_ptr"⟩] :=
  Fin.cases (fun j => f (rowsOf (vals (0 : Fin 2)) j.1.val)
    (rowsOf (vals (1 : Fin 2)) j.1.val) j.2.1.val) (fun impossible => Fin.elim0 impossible)

def backwardFormula {α : Type} {rows N : Nat} (f : α → α)
    (vals : ChannelValues α (backwardInputs rows N)) :
    ChannelValues α [⟨.real,[rows,N],"input_ptr"⟩] :=
  Fin.cases (fun j => f (vals (0 : Fin 1) j)) (fun impossible => Fin.elim0 impossible)

end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.IO

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.IO
open VeriTile.Triton FP.Structural
noncomputable section
structure LogicalForwardInputs (α : Type) (rows N : Nat) where
  prediction : Fin rows → Fin N → α
  target : Fin rows → Fin N → α

def logicalForwardInputs (α : Type) (rows N : Nat) : ChannelValues α (forwardInputs rows N) ≃ LogicalForwardInputs α rows N where
  toFun xs := ⟨fun r c => xs ⟨0,by simp [forwardInputs]⟩ (r,c,()),fun r c => xs ⟨1,by simp [forwardInputs]⟩ (r,c,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.prediction j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.target j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o <;> funext j <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def logicalBackwardInputs (α : Type) (rows N : Nat) : ChannelValues α (backwardInputs rows N) ≃ (Fin rows → Fin N → α) where
  toFun xs := fun r c => xs ⟨0,by simp [backwardInputs]⟩ (r,c,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def logicalOutputs (α : Type) (rows columns : Nat) (region : RegionName) : ChannelValues α [⟨.real,[rows,columns],region⟩] ≃ (Fin rows → Fin columns → α) where
  toFun xs := fun r c => xs ⟨0,by simp [backwardInputs]⟩ (r,c,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def extendRow {rows N : Nat} (xs : Fin rows → Fin N → ℝ) (r : Fin rows) (c : Nat) : ℝ :=
  if hc : c<N then xs r ⟨c,hc⟩ else default
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.IO

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.IO
open VeriTile.Triton FP.Structural
noncomputable section
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
def logicalForwardIO (m : Tile.Model α) (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (work : List Nat) : LogicalProgramIO α α (LogicalForwardInputs α rows N) (Fin rows → Fin (outputCount N B reduction) → α) where
  core := forward m rows sy st so N B flag reduction work
  inputs := logicalForwardInputs α rows N
  outputs := logicalOutputs α rows (outputCount N B reduction) "loss_ptr"

def logicalBackwardIO (m : Tile.Model α) (rows so st N B : Nat) (flag : Bool) (work : List Nat) : LogicalProgramIO α α (Fin rows → Fin N → α) (Fin rows → Fin N → α) where
  core := backward m rows so st N B flag work
  inputs := logicalBackwardInputs α rows N
  outputs := logicalOutputs α rows N "input_ptr"
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.IO
