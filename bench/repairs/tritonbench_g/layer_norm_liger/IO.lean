import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Logical
import VeriTile.Triton.Float.ProgramEquivalence
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.LayerNormLiger.LayerIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option autoImplicit false

def fInputs (p : Forward.Parameters) : List IOChannel :=
  [⟨.fp32,[p.block],p.x⟩,⟨.fp32,[p.block],p.weight⟩,⟨.fp32,[p.block],p.bias⟩]
def fOutputs (p : Forward.Parameters) : List IOChannel :=
  [⟨.fp32,[p.block],p.y⟩,⟨.fp32,[],p.mean⟩,⟨.fp32,[],p.rstd⟩]

/-- Actual row strides, column masks and both unmasked scalar statistics stores.
Region aliases are retained rather than rejected by the native interface. -/
def fMemory (α : Type) (p : Forward.Parameters) (pid : Nat) : MemoryIO α where
  inputs := fInputs p
  outputs := fOutputs p
  scratch := []
  buffers := [p.x,p.weight,p.bias,p.y,p.mean,p.rstd]
  read
    | ⟨0,_⟩,_,i => pid*p.xStride+i.1.val
    | ⟨1,_⟩,_,i | ⟨2,_⟩,_,i => i.1.val
  inputMask
    | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i | ⟨2,_⟩,_,i => i.1.val<p.columns
  write
    | ⟨0,_⟩,_,i => pid*p.yStride+i.1.val
    | ⟨1,_⟩,_,_ => pid*p.meanStride
    | ⟨2,_⟩,_,_ => pid*p.rstdStride
  outputMask
    | ⟨0,_⟩,_,i => i.1.val<p.columns
    | ⟨1,_⟩,_,_ | ⟨2,_⟩,_,_ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def fInvocation {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) : Invocation α where
  kernel := Forward.kernel p
  backend := withSum M R
  pids := fun _ => pid
  setPids := fun ax => ax==0

def forwardIO {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) : ProgramIO α α where
  toMemoryIO := fMemory α p pid
  schedule := fun _ => [fInvocation M R p pid]
  encode := id
  observe := fun _ _ => id

def fCandidateInvocation {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) : Invocation α := { fInvocation M R p pid with kernel := Forward.candidateKernel p }

def forwardCandidate {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) : ProgramIO.Variant (forwardIO M R p pid) where
  schedule := fun _ => [fCandidateInvocation M R p pid]
  observe := fun _ _ => id

def fInput {α : Type} (p : Forward.Parameters) (v : ChannelValues α (fInputs p)) :
    Fin 3 → Fin p.block → α
  | ⟨0,_⟩,i => v (0 : Fin 3) (i,PUnit.unit)
  | ⟨1,_⟩,i => v (1 : Fin 3) (i,PUnit.unit)
  | ⟨2,_⟩,i => v (2 : Fin 3) (i,PUnit.unit)
def fAt {α : Type} [Inhabited α] (p : Forward.Parameters) (v : ChannelValues α (fInputs p))
    (which : Fin 3) (i : Nat) : α := if h : i<p.block then fInput p v which ⟨i,h⟩ else default

/-- Independent population-statistics and affine mathematical target. -/
def forwardFormula (p : Forward.Parameters) (v : ChannelValues ℝ (fInputs p)) :
    ChannelValues ℝ (fOutputs p)
  | ⟨0,_⟩ => fun i => Logical.y p.columns p.epsilon (fAt p v 0) (fAt p v 1) (fAt p v 2) i.1.val
  | ⟨1,_⟩ => fun _ => Logical.mean p.columns (fAt p v 0)
  | ⟨2,_⟩ => fun _ => Logical.rstd p.columns p.epsilon (fAt p v 0)

def first (p : Backward.Parameters) (pid : Nat) := pid*p.rowsPerProgram
def last (p : Backward.Parameters) (pid : Nat) := min ((pid+1)*p.rowsPerProgram) p.rows
def rowMask (p : Backward.Parameters) (pid row : Nat) : Prop := first p pid≤row ∧ row<last p pid

def bInputs (p : Backward.Parameters) : List IOChannel :=
  [⟨.fp32,[p.rows,p.block],p.x⟩,⟨.fp32,[p.block],p.weight⟩,
   ⟨.fp32,[p.rows,p.block],p.dy⟩,⟨.fp32,[p.rows],p.mean⟩,⟨.fp32,[p.rows],p.rstd⟩]
def bOutputs (p : Backward.Parameters) : List IOChannel :=
  [⟨.fp32,[p.rows,p.block],p.dx⟩,⟨.fp32,[p.block],p.dw⟩,⟨.fp32,[p.block],p.db⟩]

/-- Only the executed row interval is read/written; empty loops still write
both partial-gradient vectors. All native region/stride aliases remain legal. -/
def bMemory (α : Type) (p : Backward.Parameters) (pid : Nat) : MemoryIO α where
  inputs := bInputs p
  outputs := bOutputs p
  scratch := []
  buffers := [p.x,p.weight,p.dy,p.mean,p.rstd,p.dx,p.dw,p.db]
  read
    | ⟨0,_⟩,_,i => i.1.val*p.xStride+i.2.1.val
    | ⟨1,_⟩,_,i => i.1.val
    | ⟨2,_⟩,_,i => i.1.val*p.dyStride+i.2.1.val
    | ⟨3,_⟩,_,i | ⟨4,_⟩,_,i => i.1.val
  inputMask
    | ⟨0,_⟩,_,i | ⟨2,_⟩,_,i => rowMask p pid i.1.val ∧ i.2.1.val<p.columns
    | ⟨1,_⟩,_,i => first p pid<last p pid ∧ i.1.val<p.columns
    | ⟨3,_⟩,_,i | ⟨4,_⟩,_,i => rowMask p pid i.1.val
  write
    | ⟨0,_⟩,_,i => i.1.val*p.dxStride+i.2.1.val
    | ⟨1,_⟩,_,i => pid*p.dwStride+i.1.val
    | ⟨2,_⟩,_,i => pid*p.dbStride+i.1.val
  outputMask
    | ⟨0,_⟩,_,i => rowMask p pid i.1.val ∧ i.2.1.val<p.columns
    | ⟨1,_⟩,_,i | ⟨2,_⟩,_,i => i.1.val<p.columns
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def bInvocation {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) : Invocation α where
  kernel := Backward.kernel p
  backend := withSum M R
  pids := fun _ => pid
  setPids := fun ax => ax==0

def backwardIO {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) : ProgramIO α α where
  toMemoryIO := bMemory α p pid
  schedule := fun _ => [bInvocation M R p pid]
  encode := id
  observe := fun _ _ => id

def bCandidateInvocation {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) : Invocation α := { bInvocation M R p pid with kernel := Backward.candidateKernel p }

def backwardCandidate {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) : ProgramIO.Variant (backwardIO M R p pid) where
  schedule := fun _ => [bCandidateInvocation M R p pid]
  observe := fun _ _ => id

def bx {α : Type} [Inhabited α] (p : Backward.Parameters) (v : ChannelValues α (bInputs p))
    (row col : Nat) : α := if hr : row<p.rows then if hc : col<p.block then
      v (0 : Fin 5) (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) else default else default
def bw {α : Type} [Inhabited α] (p : Backward.Parameters) (v : ChannelValues α (bInputs p))
    (col : Nat) : α := if hc : col<p.block then v (1 : Fin 5) (⟨col,hc⟩,PUnit.unit) else default
def bdy {α : Type} [Inhabited α] (p : Backward.Parameters) (v : ChannelValues α (bInputs p))
    (row col : Nat) : α := if hr : row<p.rows then if hc : col<p.block then
      v (2 : Fin 5) (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) else default else default
def bmean {α : Type} [Inhabited α] (p : Backward.Parameters) (v : ChannelValues α (bInputs p))
    (row : Nat) : α := if hr : row<p.rows then v (3 : Fin 5) (⟨row,hr⟩,PUnit.unit) else default
def brstd {α : Type} [Inhabited α] (p : Backward.Parameters) (v : ChannelValues α (bInputs p))
    (row : Nat) : α := if hr : row<p.rows then v (4 : Fin 5) (⟨row,hr⟩,PUnit.unit) else default

def dwFormula (p : Backward.Parameters) (pid : Nat) (v : ChannelValues ℝ (bInputs p))
    (col : Nat) : ℝ := ∑ row ∈ Finset.Ico (first p pid) (last p pid),
  bdy p v row col*Logical.xhat (bx p v row) (bmean p v row) (brstd p v row) col
def dbFormula (p : Backward.Parameters) (pid : Nat) (v : ChannelValues ℝ (bInputs p))
    (col : Nat) : ℝ := ∑ row ∈ Finset.Ico (first p pid) (last p pid),bdy p v row col

/-- Independent DX and sums of the two partial parameter gradients over the
program's actual row interval, using the saved forward statistics as inputs. -/
def backwardFormula (p : Backward.Parameters) (pid : Nat) (v : ChannelValues ℝ (bInputs p)) :
    ChannelValues ℝ (bOutputs p)
  | ⟨0,_⟩ => fun i => Logical.dx p.columns (bx p v i.1.val) (bw p v) (bdy p v i.1.val)
      (bmean p v i.1.val) (brstd p v i.1.val) i.2.1.val
  | ⟨1,_⟩ => fun i => dwFormula p pid v i.1.val
  | ⟨2,_⟩ => fun i => dbFormula p pid v i.1.val

end
end VeriTile.Bench.Repairs.LayerNormLiger.LayerIO

namespace VeriTile.Bench.Repairs.LayerNormLiger.LogicalBackwardIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (p : Backward.Parameters) where
  x : Fin p.rows → Fin p.block → ℝ
  weight : Fin p.block → ℝ
  dy : Fin p.rows → Fin p.block → ℝ
  mean : Fin p.rows → ℝ
  inverse : Fin p.rows → ℝ

def inputs (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat) : (LayerIO.backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Inputs ≃ Inputs p where
  toFun := fun v => ⟨fun i0 i1 => v (0 : Fin 5) (i0,i1,()), fun i0 => v (1 : Fin 5) (i0,()), fun i0 i1 => v (2 : Fin 5) (i0,i1,()), fun i0 => v (3 : Fin 5) (i0,()), fun i0 => v (4 : Fin 5) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.x j.1 j.2.1
    | ⟨1,_⟩ => fun j => v.weight j.1
    | ⟨2,_⟩ => fun j => v.dy j.1 j.2.1
    | ⟨3,_⟩ => fun j => v.mean j.1
    | ⟨4,_⟩ => fun j => v.inverse j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (p : Backward.Parameters) where
  dx : Fin p.rows → Fin p.block → ℝ
  dw : Fin p.block → ℝ
  db : Fin p.block → ℝ

def outputs (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat) : (LayerIO.backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Outputs ≃ Outputs p where
  toFun := fun v => ⟨fun i0 i1 => v (0 : Fin 3) (i0,i1,()), fun i0 => v (1 : Fin 3) (i0,()), fun i0 => v (2 : Fin 3) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.dx j.1 j.2.1
    | ⟨1,_⟩ => fun j => v.dw j.1
    | ⟨2,_⟩ => fun j => v.db j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat) : LogicalProgramIO ℝ ℝ (Inputs p) (Outputs p) where
  core := LayerIO.backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid
  inputs := inputs rest p pid
  outputs := outputs rest p pid


def extend {n : Nat} (x : Fin n → ℝ) (i : Nat) : ℝ := if h : i<n then x ⟨i,h⟩ else 0

def extend2 {m n : Nat} (x : Fin m → Fin n → ℝ) (i j : Nat) : ℝ :=
  if hi : i<m then if hj : j<n then x ⟨i,hi⟩ ⟨j,hj⟩ else 0 else 0

end
end VeriTile.Bench.Repairs.LayerNormLiger.LogicalBackwardIO

namespace VeriTile.Bench.Repairs.LayerNormLiger.LogicalForwardIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (p : Forward.Parameters) where
  x : Fin p.block → ℝ
  weight : Fin p.block → ℝ
  bias : Fin p.block → ℝ

def inputs (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat) : (LayerIO.forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Inputs ≃ Inputs p where
  toFun := fun v => ⟨fun i0 => v (0 : Fin 3) (i0,()), fun i0 => v (1 : Fin 3) (i0,()), fun i0 => v (2 : Fin 3) (i0,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.x j.1
    | ⟨1,_⟩ => fun j => v.weight j.1
    | ⟨2,_⟩ => fun j => v.bias j.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (p : Forward.Parameters) where
  y : Fin p.block → ℝ
  mean : ℝ
  inverse : ℝ

def outputs (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat) : (LayerIO.forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Outputs ≃ Outputs p where
  toFun := fun v => ⟨fun i0 => v (0 : Fin 3) (i0,()), v (1 : Fin 3) (), v (2 : Fin 3) ()⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.y j.1
    | ⟨1,_⟩ => fun j => v.mean 
    | ⟨2,_⟩ => fun j => v.inverse 
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat) : LogicalProgramIO ℝ ℝ (Inputs p) (Outputs p) where
  core := LayerIO.forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid
  inputs := inputs rest p pid
  outputs := outputs rest p pid


def extend {n : Nat} (x : Fin n → ℝ) (i : Nat) : ℝ := if h : i<n then x ⟨i,h⟩ else 0

def extend2 {m n : Nat} (x : Fin m → Fin n → ℝ) (i j : Nat) : ℝ :=
  if hi : i<m then if hj : j<n then x ⟨i,hi⟩ ⟨j,hj⟩ else 0 else 0

end
end VeriTile.Bench.Repairs.LayerNormLiger.LogicalForwardIO
