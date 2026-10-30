import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelSource
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.IO
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

abbrev channels (N M : Nat) : List IOChannel :=
  [⟨.fp32, [N, M], "A"⟩, ⟨.fp32, [M], "B"⟩]
abbrev Inputs (α : Type) (N M : Nat) := ChannelValues α (channels N M)

def matrixValues {α : Type} [Inhabited α] {N M : Nat}
    (vals : Inputs α N M) (i j : Nat) : α :=
  if h : i < N ∧ j < M then vals 0 (⟨i, h.1⟩, ⟨j, h.2⟩, PUnit.unit) else default

def vectorValues {α : Type} [Inhabited α] {N M : Nat}
    (vals : Inputs α N M) (j : Nat) : α :=
  if h : j < M then vals 1 (⟨j, h⟩, PUnit.unit) else default

def invocation {β : Type} (A : Algebra β) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) : Invocation β where
  kernel := Kernels.source N M BN BM rowShape columnShape
  backend := A
  pids := fun ax => if ax = 0 then pid else 0

/-- Common row-major input windows for the single-program and whole-grid views. -/
abbrev inputRead {α : Type} (N M : Nat) :
    (i : Fin (channels N M).length) → Inputs α N M → TileIndex (channels N M)[i].shape → Nat :=
  Fin.cases (fun _ j => j.1.val * M + j.2.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => i.elim0))

/-- Shared typed layout used by the independently displayed Real/native IO.
It retains the entire matrix/vector inputs and the one program's output rows. -/
abbrev io {α β : Type} (A : Algebra β) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := channels N M
  outputs := [⟨.fp32, [BN], "C"⟩]
  scratch := []
  buffers := ["A", "B", "C"]
  read := inputRead N M
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ i => pid * BN + i.1.val) (fun i => i.elim0)
  outputMask := Fin.cases (fun _ i => pid * BN + i.1.val < N) (fun i => i.elim0)
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [invocation A N M BN BM pid rowShape columnShape]
  encode := encode
  observe := fun _ _ => observe

/-- A single logical result vector in the public typed output channel. -/
def outputValues {α : Type} {BN : Nat} (f : Fin BN → α) :
    ChannelValues α [⟨.fp32, [BN], "C"⟩] :=
  Fin.cases (fun i => f i.1) (fun i => i.elim0)

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.IO

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.IO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Both complete logical tensors, including every input outside this PID's output tile. -/
def logicalInputs (N M : Nat) :
    Inputs ℝ N M ≃ ((Fin N → Fin M → ℝ) × (Fin M → ℝ)) where
  toFun := fun xs => (fun row col => xs 0 (row,col,PUnit.unit), fun col => xs 1 (col,PUnit.unit))
  invFun := fun av => Fin.cases (fun j => av.1 j.1 j.2.1)
    (Fin.cases (fun j => av.2 j.1) (fun port => Fin.elim0 port))
  left_inv := by
    intro xs
    funext port j
    fin_cases port
    · rcases j with ⟨row,col,⟨⟩⟩; rfl
    · rcases j with ⟨col,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

/-- The complete original BN output tile; its active row mask remains in core IO. -/
def logicalOutputs (BN : Nat) :
    ChannelValues ℝ [⟨.fp32,[BN],"C"⟩] ≃ (Fin BN → ℝ) where
  toFun := fun ys row => ys 0 (row,PUnit.unit)
  invFun := outputValues
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨row,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.IO

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.IO
open VeriTile.Triton FP.Structural

noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (A : Algebra ℝ) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) : LogicalProgramIO ℝ ℝ
    ((Fin N → Fin M → ℝ) × (Fin M → ℝ)) (Fin BN → ℝ) where
  core := io A N M BN BM pid rowShape columnShape id id
  inputs := logicalInputs N M
  outputs := logicalOutputs BN

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.IO

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.IO
open VeriTile.Triton
noncomputable section

/-- Logical row lookup for a tail-masked output tile. Outside the finite
matrix it has exactly the old irrelevant zero extension. -/
def logicalRow {N M : Nat} (matrix : Fin N → Fin M → ℝ) (row : Nat) (col : Fin M) : ℝ :=
  if h : row<N then matrix ⟨row,h⟩ col else 0

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.IO
