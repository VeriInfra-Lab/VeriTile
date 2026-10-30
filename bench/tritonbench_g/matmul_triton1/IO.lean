import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.matmul_triton1.Kernels
import bench.tritonbench_g.matmul_triton1.Proofs.KernelInterface
import VeriTile.Triton.Float.ProgramIO

/-! Logical matrices, exact source layout and complete row-major launch schedule.
The Structural `.real` cell channel carries the source's FP32 payloads; the
unchanged typed quotation fixes loads, live dot carry and stores to FP32. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

def inputs (p : Core.Parameters) : List IOChannel :=
  [⟨.real,[p.rows,p.innerSize],p.input⟩,⟨.real,[p.innerSize,p.columns],p.weight⟩]
def outputs (p : Core.Parameters) : List IOChannel :=
  [⟨.real,[p.rows,p.columns],p.output⟩]

def memoryIO (α : Type) (p : Core.Parameters) : MemoryIO α where
  inputs := inputs p
  outputs := outputs p
  buffers := [p.input,p.weight,p.output]
  read := Fin.cases (fun _ j => j.1.val*p.inputRow+j.2.1.val*p.inputInner)
    (Fin.cases (fun _ j => j.1.val*p.weightInner+j.2.1.val*p.weightColumn)
      (fun i => Fin.elim0 i))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*p.outputRow+j.2.1.val*p.outputColumn)
    (fun i => Fin.elim0 i)
  outputMask := fun _ _ _ => True
  scratchWrite := by intro i; exact Fin.elim0 i
  scratchMask := by intro i; exact Fin.elim0 i

def launch {α : Type} (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (pid : Nat) : Invocation α where
  kernel := Kernel.source p
  backend := Core.backend rest P
  pids := fun ax => if ax=0 then pid else 0

def io {α : Type} (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) : ProgramIO α α where
  toMemoryIO := memoryIO α p
  schedule := fun _ => (List.range (Geometry.count (Core.geometry p))).map (launch rest P p)
  encode := id
  observe := fun _ _ => id

/-- Logical K padding is explicit: native dots retain every padded zero lane. -/
def left {α : Type} (p : Core.Parameters) (xs : ChannelValues α (inputs p))
    (zero : α) (r : Fin p.rows) (k : Nat) : α :=
  if h : k<p.innerSize then xs (0 : Fin 2) (r,⟨k,h⟩,()) else zero
def right {α : Type} (p : Core.Parameters) (xs : ChannelValues α (inputs p))
    (zero : α) (c : Fin p.columns) (k : Nat) : α :=
  if h : k<p.innerSize then xs (1 : Fin 2) (⟨k,h⟩,c,()) else zero

/-- Matrix product over exactly the original K extent, independent of blocks,
source evaluation, launch ordering and intermediate memory. -/
def realTarget (p : Core.Parameters) (xs : ChannelValues ℝ (inputs p)) :
    ChannelValues ℝ (outputs p) :=
  Fin.cases (fun j => RealSum.sum
    (fun k => left p xs 0 j.1 k * right p xs 0 j.2.1 k) p.innerSize)
    (fun i => Fin.elim0 i)

/-- Original ordered block-dot recurrence, parameterized by the explicit
primitive model. Native instantiation uses ascending scalar FP32 FMA lanes. -/
def precisionAccum {α : Type} (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (r : Fin p.rows) (c : Fin p.columns) : Nat → α
  | 0 => P.zero
  | block+1 => P.dot
      (fun k : Fin p.blockInner => left p xs P.zero r (block*p.blockInner+k.val))
      (fun k : Fin p.blockInner => right p xs P.zero c (block*p.blockInner+k.val))
      (precisionAccum P p xs r c block)

def precisionTarget {α : Type} (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) : ChannelValues α (outputs p) :=
  Fin.cases (fun j => precisionAccum P p xs j.1 j.2.1 (Core.steps p))
    (fun i => Fin.elim0 i)

end
end VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO

namespace VeriTile.Bench.Repairs.MatmulTriton1.Pair
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
variable {α : Type} [Inhabited α]

def launch (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) (pid : Nat) : Invocation α :=
  { MatmulIO.launch rest P p pid with kernel := Kernel.maskHoisted p }

def candidateIO (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) : ProgramIO α α :=
  { MatmulIO.io rest P p with schedule := fun _ =>
      (List.range (Geometry.count (Core.geometry p))).map (launch rest P p) }

def variant (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :
    ProgramIO.Variant (MatmulIO.io rest P p) where
  schedule := (candidateIO rest P p).schedule
  observe := (MatmulIO.io rest P p).observe

end
end VeriTile.Bench.Repairs.MatmulTriton1.Pair

namespace VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete input channels reindexed as two ordinary logical matrices. -/
def matrixInputs (p : Revision001.Core.Parameters) :
    ChannelValues ℝ (inputs p) ≃ ((Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ)) where
  toFun := fun xs => (fun row col => id (xs (0 : Fin 2) (row,col,PUnit.unit)),
    fun row col => id (xs (1 : Fin 2) (row,col,PUnit.unit)))
  invFun := fun ab => Fin.cases (fun j => id (ab.1 j.1 j.2.1))
    (Fin.cases (fun j => id (ab.2 j.1 j.2.1)) (fun port => Fin.elim0 port))
  left_inv := by
    intro xs
    funext port j
    fin_cases port <;> rcases j with ⟨row,col,⟨⟩⟩ <;> rfl
  right_inv := by
    intro ab
    apply Prod.ext <;> funext row col <;> rfl

/-- Complete sole output channel reindexed as an ordinary logical matrix. -/
def matrixOutputs (p : Revision001.Core.Parameters) :
    ChannelValues ℝ (outputs p) ≃ (Fin p.rows → Fin p.columns → ℝ) where
  toFun := fun ys row col => id (ys (0 : Fin 1) (row,col,PUnit.unit))
  invFun := fun f => Fin.cases (fun j => id (f j.1 j.2.1)) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨row,col,⟨⟩⟩
    rfl
  right_inv := by
    intro f
    funext row col
    rfl

end
end VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO

namespace VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO
open VeriTile.Triton FP.Structural

noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (rest : Algebra ℝ) (P : Revision001.Core.Primitives ℝ) (p : Revision001.Core.Parameters) : LogicalProgramIO ℝ ℝ
    ((Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ)) (Fin p.rows → Fin p.columns → ℝ) where
  core := io rest P p
  inputs := matrixInputs p
  outputs := matrixOutputs p

end
end VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO
