import bench.tritonbench_g.matmul_triton1.Proofs.FPComplete
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- Logical operands depend only on input memory, logical matrix indices and
the original K extent. Padded K lanes remain explicit zeros for FP semantics. -/
def logicalA (P : Primitives α) (p : Parameters) (s : State α) (r block : Nat)
    (k : Fin p.blockInner) : α :=
  if k.val< inner p-block*p.blockInner then
    (s.mem p.input (r*p.inputRow+(block*p.blockInner+k.val)*p.inputInner)).read .real
  else P.zero
def logicalB (P : Primitives α) (p : Parameters) (s : State α) (c block : Nat)
    (k : Fin p.blockInner) : α :=
  if k.val< inner p-block*p.blockInner then
    (s.mem p.weight ((block*p.blockInner+k.val)*p.weightInner+c*p.weightColumn)).read .real
  else P.zero
def logicalAccum (P : Primitives α) (p : Parameters) (s : State α) (r c : Nat) : Nat → α
  | 0 => P.zero
  | block+1 => P.dot (logicalA P p s r block) (logicalB P p s c block)
      (logicalAccum P p s r c block)
def logical (P : Primitives α) (p : Parameters) (s : State α) (r c : Nat) : α :=
  logicalAccum P p s r c (steps p)

theorem a_logical (P : Primitives α) (p : Parameters) (s : State α) (block : Nat)
    (i : Fin p.blockRows) (j : Fin p.blockInner) (hi : row p s i<p.rows) :
    aValues P p s block (i,j,PUnit.unit)=logicalA P p s (row p s i) block j := by
  have offset : aOffset p s block (i,j,PUnit.unit)=
      row p s i*p.inputRow+(block*p.blockInner+j.val)*p.inputInner := by
    simp only [aOffset]; ring
  simp [aValues,logicalA,hi,offset]
theorem b_logical (P : Primitives α) (p : Parameters) (s : State α) (block : Nat)
    (i : Fin p.blockInner) (j : Fin p.blockColumns) (hj : column p s j<p.columns) :
    bValues P p s block (i,j,PUnit.unit)=logicalB P p s (column p s j) block i := by
  have offset : bOffset p s block (i,j,PUnit.unit)=
      (block*p.blockInner+i.val)*p.weightInner+column p s j*p.weightColumn := by
    simp only [bOffset]; ring
  simp [bValues,logicalB,hj,offset]
theorem accumulation_logical (P : Primitives α) (p : Parameters) (s : State α)
    (i : CIndex p) (hi : active p s i) (n : Nat) :
    accumulation P p s n i=logicalAccum P p s (row p s i.1) (column p s i.2.1) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [accumulation,matrixDot,logicalAccum,ih]
    congr 1
    · funext j; exact a_logical P p s n i.1 j hi.1
    · funext j; exact b_logical P p s n j i.2.1 hi.2

theorem logical_preserved (P : Primitives α) (p : Parameters) (initial current : State α)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hout : GridExecution.Outside p.output initial current) (r c : Nat) :
    logical P p current r c=logical P p initial r c := by
  have h : ∀ n,logicalAccum P p current r c n=logicalAccum P p initial r c n := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      simp only [logicalAccum,ih]
      congr 1
      · funext k; simp only [logicalA,hout p.input _ ha]
      · funext k; simp only [logicalB,hout p.weight _ hb]
  exact h _

#specNonCircular logical avoiding [source,FP.Structural.exec]
#axiomsClean a_logical
#axiomsClean b_logical
#axiomsClean accumulation_logical
#axiomsClean logical_preserved
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
