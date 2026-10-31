import bench.tritonbench_g.matmul_triton1.Proofs.FPSource
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
noncomputable section

def blocksM (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def rowBlock {α : Type} (p : Parameters) (s : State α) := s.pids 0/blocksN p
def columnBlock {α : Type} (p : Parameters) (s : State α) := s.pids 0%blocksN p
def row {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockRows) := rowBlock p s*p.blockRows+i.val
def column {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockColumns) := columnBlock p s*p.blockColumns+j.val

def aOffset {α : Type} (p : Parameters) (s : State α) (block : Nat) (i : AIndex p) :=
  row p s i.1*p.inputRow+i.2.1.val*p.inputInner+block*(p.blockInner*p.inputInner)
def bOffset {α : Type} (p : Parameters) (s : State α) (block : Nat) (i : BIndex p) :=
  i.1.val*p.weightInner+column p s i.2.1*p.weightColumn+block*(p.blockInner*p.weightInner)
def cOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) :=
  row p s i.1*p.outputRow+column p s i.2.1*p.outputColumn

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  row p s i.1<p.rows ∧ column p s i.2.1<p.columns
instance {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Decidable (active p s i) := inferInstanceAs (Decidable (_ ∧ _))

def aValues {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α)
    (block : Nat) (i : AIndex p) : α :=
  if row p s i.1 < p.rows ∧ i.2.1.val < inner p-block*p.blockInner then (s.mem p.input (aOffset p s block i)).read .real else P.zero
def bValues {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α)
    (block : Nat) (i : BIndex p) : α :=
  if i.1.val < inner p-block*p.blockInner ∧ column p s i.2.1 < p.columns then (s.mem p.weight (bOffset p s block i)).read .real else P.zero

/-- The source's masked operands and fused dot-accumulator, from
initial memory only. No masked product or explicit zero seed is discarded. -/
def accumulation {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters)
    (initial : State α) : Nat → CIndex p → α
  | 0 => fun _ => P.zero
  | block+1 => matrixDot P (batch:=[]) (aValues P p initial block) (bValues P p initial block)
      (accumulation P p initial block)

#specNonCircular accumulation avoiding [source, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
