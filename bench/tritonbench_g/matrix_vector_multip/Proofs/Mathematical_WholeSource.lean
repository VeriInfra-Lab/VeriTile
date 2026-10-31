import bench.tritonbench_g.matrix_vector_multip.Sources.Parameters
import VeriTile.Triton.Float.Control
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeSource
open VeriTile.Triton FP.Structural Typed
noncomputable section
set_option maxRecDepth 24000
/-- Pure operand geometry and ordered mathematical recurrence. There is no alternative kernel here. -/
abbrev Acc (α : Type) (p : Parameters) := Values α .real [p.blockRows,p.blockColumns]
def row (p : Parameters) (pid i : Nat) := pid*p.blockRows+i
def aOffset (p : Parameters) (pid off i j : Nat) :=
  row p pid i*p.strideRow+j*p.strideColumn+off*p.strideColumn
def bOffset (p : Parameters) (off j : Nat) := j*p.strideVector+off*p.strideVector

def Context {α : Type} (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α p) (s : State α) : Prop :=
  s.regs .nat [p.blockRows,1] "offset_n"=some (fun i => row p pid i.1.val) ∧
  s.regs .nat [1,p.blockColumns] "offset_m"=some (fun i => i.2.1.val) ∧
  s.regs .bool [p.blockRows,1] "n_mask"=some (fun i => decide (row p pid i.1.val<p.rows)) ∧
  s.regs .ptr [p.blockRows,p.blockColumns] "A_ptrs"=
    some (fun i => (p.a,aOffset p pid off i.1.val i.2.1.val)) ∧
  s.regs .ptr [1,p.blockColumns] "B_ptrs"=some (fun i => (p.b,bOffset p off i.2.1.val)) ∧
  s.regs .real [p.blockRows,p.blockColumns] "acc"=some acc ∧ s.mem=memory

def load {α : Type} [Inhabited α] (A : Algebra α) (memory : RegionName → Nat → Cell α)
    (region : RegionName) (offset : Nat) (active : Prop) [Decidable active] : α :=
  A.cast (some .fp32) .fp32 .real (if active then (memory region offset).read .fp32
    else A.cast (some .fp32) .real .fp32 (A.literal (some .fp32) .real 0))

def aChunk {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) : Acc α p :=
  fun i => load A memory p.a (aOffset p pid off i.1.val i.2.1.val)
    (row p pid i.1.val<p.rows ∧ off+i.2.1.val<p.columns)
def bChunk {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (off : Nat)
    (memory : RegionName → Nat → Cell α) : Values α .real [1,p.blockColumns] :=
  fun i => load A memory p.b (bOffset p off i.2.1.val) (off+i.2.1.val<p.columns)
def nextAcc {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α p) : Acc α p :=
  fun i => A.binary (some .fp32) .real .add (acc i)
    (A.binary (some .fp32) .real .mul (aChunk A p pid off memory i)
      (bChunk A p off memory (⟨0,by omega⟩,i.2.1,PUnit.unit)))

def accumulated {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid : Nat)
    (memory : RegionName → Nat → Cell α) (off : Nat) (acc : Acc α p) : Acc α p :=
  if h : off<p.columns ∧ 0<p.blockColumns then
    accumulated A p pid memory (off+p.blockColumns) (nextAcc A p pid off memory acc)
  else acc
termination_by p.columns-off

def outputValues {α : Type} (A : Algebra α) (p : Parameters) (acc : Acc α p) :
    Values α .real [p.blockRows] :=
  A.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false acc

def storedValue {α : Type} (A : Algebra α) (p : Parameters) (acc : Acc α p)
    (i : Fin p.blockRows) : α :=
  A.cast none .real .fp32 (outputValues A p acc (i,PUnit.unit))

theorem row_injective (p : Parameters) (pid : Nat)
    (stride : p.strideOut=1) :
    Function.Injective (fun i : TileIndex [p.blockRows,1] => row p pid i.1.val*p.strideOut) := by
  rintro ⟨a,b,u⟩ ⟨c,d,v⟩ he
  have hac : a=c := Fin.ext (by simpa [row,stride] using he)
  subst c
  have hbd : b=d := Subsingleton.elim _ _
  subst d
  cases u; cases v; rfl

def tileValue {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid : Nat)
    (memory : RegionName → Nat → Cell α) (i : Fin p.blockRows) : α :=
  storedValue A p (accumulated A p pid memory 0 (fun _ => A.literal (some .fp32) .real 0)) i

end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeSource
