import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.GraphResults
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.RawGridProof
import VeriTile.Triton.Float.PerAddressProgramIO

namespace VeriTile.Bench.Repairs.Int4Matmul.IO
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section

/-- One complete CTA, including both its grouped matrix tile and split-K axis. -/
abbrev Program := Nat × Nat

def programs (s : Shape) (t : Tile) : List Program :=
  (List.range (blocksM s t * blocksN s t)).product (List.range t.splits)

def owns (s : Shape) (t : Tile) (p : Program) (row col : Nat) : Prop :=
  row / t.bm = GroupedGrid.row (blocksM s t) (blocksN s t) t.gm p.1 ∧
  col / t.bn = GroupedGrid.col (blocksM s t) (blocksN s t) t.gm p.1

instance (s : Shape) (t : Tile) (p : Program) (row col : Nat) : Decidable (owns s t p row col) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- Canonical contributing CTAs; all other CTAs remain in the launch catalog. -/
def events (s : Shape) (t : Tile) (row col : Nat) : List Program :=
  (programs s t).filter (fun p => decide (owns s t p row col))

/-- Project the original arbitrary complete lane serialization separately at
 each output address. This does not impose a common CTA order on all outputs. -/
def order (s : Shape) (t : Tile) (lanes : List Lane) (row col : Nat) : List Program :=
  (lanes.filter (fun l => decide ((laneWork s t l).row=row ∧
    (laneWork s t l).col=col))).map (fun l => (l.pid,l.split))

@[reducible] def inputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.m*s.k],"a"⟩, ⟨.int,[(s.k/8)*s.n],"weight"⟩,
   ⟨.real,[(s.k/s.group)*s.n],"scale"⟩,
   ⟨.int,[(s.k/s.group)*(s.n/8)],"zero"⟩,
   ⟨.real,[s.m*s.n],"out"⟩]
@[reducible] def outputs (s : Shape) : List IOChannel := [⟨.real,[s.m,s.n],"out"⟩]

/-- The source retains FP32 A/scale/output and INT32 packed loads. Structural
`.real` is the payload carrier; source precision remains in KernelSource.kernel. -/
def invocation (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (p : Program) : Invocation (SourceGraph.Node α) where
  kernel := KernelSource.kernel (Raw.parameters x t)
  backend := SourceGraph.engine o rest
  arguments := []
  clearRegisters := Bool.false
  pids := fun ax => if ax=0 then p.1 else if ax=1 then p.2 else 0
  setPids := fun ax => decide (ax=0 ∨ ax=1)
  numPids := none

/-- Finite operand windows, exact in-place output reset boundary, all complete
CTAs and the original per-address serialization. No private scratch window. -/
@[reducible] def programIO (x : Raw.Arguments) (t : Tile) (o : Ops α)
    (rest : Algebra α) (lanes : List Lane) :
    PerAddressProgramIO α (SourceGraph.Node α) α Program where
  inputs := inputs x.shape
  outputs := outputs x.shape
  scratch := []
  buffers := ["a","weight","scale","zero","out"]
  read := Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => j.1.val) (Fin.cases (fun _ j => j.1.val)
      (Fin.cases (fun _ j => j.1.val) (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i)))))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*x.shape.n+j.2.1.val) (fun i => Fin.elim0 i)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  programs := fun _ => programs x.shape t
  invocation := fun _ => invocation x t o rest
  events := fun _ => Fin.cases (fun j => events x.shape t j.1.val j.2.1.val) (fun i => Fin.elim0 i)
  order := fun _ => Fin.cases (fun j => order x.shape t lanes j.1.val j.2.1.val) (fun i => Fin.elim0 i)
  encode := SourceGraph.Node.value
  observe := fun _ _ => SourceGraph.observe o

/-- Only packed signed INT32 interpretation and the original zero-initialized
output boundary. All floating values, including exceptional native words, are admitted. -/
def domain (s : Shape) (o : Ops α) (vals : ChannelValues α (inputs s)) : Prop :=
  (∀ i : Fin ((s.k/8)*s.n), (BitVec.ofInt 32 (vals 1 (i,()))).toInt=vals 1 (i,())) ∧
  (∀ i : Fin ((s.k/s.group)*(s.n/8)), (BitVec.ofInt 32 (vals 3 (i,()))).toInt=vals 3 (i,())) ∧
  (∀ i : Fin (s.m*s.n), vals 4 (i,())=o.zero)
end
end VeriTile.Bench.Repairs.Int4Matmul.IO

namespace VeriTile.Bench.Repairs.Int4Matmul.IO
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section

/-- Genuine matrix coordinates for every cell of a packed flat operand. -/
def matrixValues (α : Type) (dtype : TileDType) (m n : Nat) :
    Values α dtype [m*n] ≃ (Fin m → Fin n → Value α dtype) where
  toFun := fun xs r c => xs (finProdFinEquiv (r,c),())
  invFun := fun xs j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs
    funext j
    change xs (finProdFinEquiv (finProdFinEquiv.symm j.1),()) = xs j
    rw [Equiv.apply_symm_apply]
  right_inv := by
    intro xs
    funext r c
    change (let j := finProdFinEquiv.symm (finProdFinEquiv (r,c)); xs j.1 j.2) = xs r c
    rw [Equiv.symm_apply_apply]

structure MatrixInputs (α : Type) (s : Shape) where
  a : Fin s.m → Fin s.k → α
  weight : Fin (s.k/8) → Fin s.n → Int
  scale : Fin (s.k/s.group) → Fin s.n → α
  zero : Fin (s.k/s.group) → Fin (s.n/8) → Int
  initialOutput : Fin s.m → Fin s.n → α

def inputCoordinates (α : Type) (s : Shape) : ChannelValues α (inputs s) ≃ MatrixInputs α s where
  toFun := fun xs => ⟨matrixValues α .real s.m s.k (xs 0),
    matrixValues α .int (s.k/8) s.n (xs 1),matrixValues α .real (s.k/s.group) s.n (xs 2),
    matrixValues α .int (s.k/s.group) (s.n/8) (xs 3),matrixValues α .real s.m s.n (xs 4)⟩
  invFun := fun xs => Fin.cases ((matrixValues α .real s.m s.k).symm xs.a)
    (Fin.cases ((matrixValues α .int (s.k/8) s.n).symm xs.weight)
      (Fin.cases ((matrixValues α .real (s.k/s.group) s.n).symm xs.scale)
        (Fin.cases ((matrixValues α .int (s.k/s.group) (s.n/8)).symm xs.zero)
          (Fin.cases ((matrixValues α .real s.m s.n).symm xs.initialOutput) (fun i => Fin.elim0 i)))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (matrixValues α .real s.m s.k).symm_apply_apply (xs 0)
    · exact (matrixValues α .int (s.k/8) s.n).symm_apply_apply (xs 1)
    · exact (matrixValues α .real (s.k/s.group) s.n).symm_apply_apply (xs 2)
    · exact (matrixValues α .int (s.k/s.group) (s.n/8)).symm_apply_apply (xs 3)
    · exact (matrixValues α .real s.m s.n).symm_apply_apply (xs 4)
  right_inv := by
    intro xs
    change MatrixInputs.mk
      ((matrixValues α .real s.m s.k) ((matrixValues α .real s.m s.k).symm xs.a))
      ((matrixValues α .int (s.k/8) s.n) ((matrixValues α .int (s.k/8) s.n).symm xs.weight))
      ((matrixValues α .real (s.k/s.group) s.n) ((matrixValues α .real (s.k/s.group) s.n).symm xs.scale))
      ((matrixValues α .int (s.k/s.group) (s.n/8)) ((matrixValues α .int (s.k/s.group) (s.n/8)).symm xs.zero))
      ((matrixValues α .real s.m s.n) ((matrixValues α .real s.m s.n).symm xs.initialOutput)) = xs
    simp only [Equiv.apply_symm_apply]

def outputCoordinates (s : Shape) : ChannelValues ℝ (outputs s) ≃ (Fin s.m → Fin s.n → ℝ) :=
  (ChannelValues.singleEquiv ℝ ⟨.real,[s.m,s.n],"out"⟩).trans (Values.matrixEquiv ℝ .real s.m s.n)

/-- Canonical signed packed INT32 words and exactly the existing zero reset.
The initial output remains an input of the bijection, including all its cells. -/
def matrixDomain (s : Shape) (xs : MatrixInputs ℝ s) : Prop :=
  domain s realOps ((inputCoordinates ℝ s).symm xs)

def keyPack (s : Shape) (h : s.Valid) (k : Fin s.k) : Fin (s.k/8) :=
  ⟨k.val/8, (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    (by simpa only [Nat.div_mul_cancel h.k8] using k.isLt)⟩

def groupIndex (s : Shape) (h : s.Valid) (k : Fin s.k) : Fin (s.k/s.group) :=
  ⟨k.val/s.group, (Nat.div_lt_iff_lt_mul h.gpos).mpr
    (by simpa only [Nat.div_mul_cancel h.groups] using k.isLt)⟩

def columnPack (s : Shape) (h : s.Valid) (c : Fin s.n) : Fin (s.n/8) :=
  ⟨c.val/8, (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    (by simpa only [Nat.div_mul_cancel h.n8] using c.isLt)⟩

def weightNibble (s : Shape) (h : s.Valid) (xs : MatrixInputs ℝ s)
    (k : Fin s.k) (c : Fin s.n) : Int :=
  nibble (BitVec.ofInt 32 (xs.weight (keyPack s h k) c)) k.val

def zeroNibble (s : Shape) (h : s.Valid) (xs : MatrixInputs ℝ s)
    (k : Fin s.k) (c : Fin s.n) : Int :=
  nibble (BitVec.ofInt 32 (xs.zero (groupIndex s h k) (columnPack s h c))) c.val

/-- The unchanged per-address source judgment; each output keeps its own order. -/
def logicalIO (x : Raw.Arguments) (t : Tile) (rest : Algebra ℝ) (lanes : List Lane) :
    LogicalDomainIO (ChannelValues ℝ (inputs x.shape)) (ChannelValues ℝ (outputs x.shape))
      (MatrixInputs ℝ x.shape) (Fin x.shape.m → Fin x.shape.n → ℝ)
      (IO.programIO x t realOps rest lanes).ImplementsOnCells where
  inputs := inputCoordinates ℝ x.shape
  outputs := outputCoordinates x.shape
end
end VeriTile.Bench.Repairs.Int4Matmul.IO
