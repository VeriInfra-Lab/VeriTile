import bench.tritonbench_g.matmul_triton1.Proofs.Grid
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteOperands
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001

noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Canonical (p : Core.Parameters) : Prop where
  inputRow : p.inputRow=p.innerSize
  inputInner : p.inputInner=1
  weightInner : p.weightInner=p.columns
  weightColumn : p.weightColumn=1
  outputRow : p.outputRow=p.columns
  outputColumn : p.outputColumn=1

theorem a_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) (k : Nat) :
    FiniteOperands.naturalA P p Bool.false s k = Core.aValues P p s k := by
  funext i
  unfold FiniteOperands.naturalA Core.aValues
  change (if Core.row p s i.1 < p.rows ∧ i.2.1.val < Core.inner p-k*p.blockInner then
    (s.mem p.input (Core.row p s i.1*p.innerSize+(k*p.blockInner+i.2.1.val))).read .real else _) = _
  simp only [Core.aOffset,h.inputRow,h.inputInner,Nat.mul_one,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem b_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) (k : Nat) :
    FiniteOperands.naturalB P p Bool.false s k = Core.bValues P p s k := by
  funext i
  unfold FiniteOperands.naturalB Core.bValues
  change (if i.1.val < Core.inner p-k*p.blockInner ∧ Core.column p s i.2.1 < p.columns then
    (s.mem p.weight ((k*p.blockInner+i.1.val)*p.columns+Core.column p s i.2.1)).read .real else _) = _
  simp only [Core.bOffset,h.weightInner,h.weightColumn,Nat.mul_one,Nat.add_mul,
    Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem accumulation_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) (k : Nat) :
    FiniteOperands.accumulation P p Bool.false s k = Core.accumulation P p s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [FiniteOperands.accumulation,Core.accumulation,a_eq P p h,b_eq P p h,ih]

theorem written_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) : FiniteOperands.written P p Bool.false s = Core.written P p s := by
  unfold FiniteOperands.written Core.written
  congr 1
  funext t i
  have active : FiniteOperands.active p Bool.false s i = Core.active p s i := rfl
  have offset : FiniteOperands.offset p Bool.false s i = Core.cOffset p s i := by
    change Core.row p s i.1*p.columns+Core.column p s i.2.1 = _
    simp [Core.cOffset,h.outputRow,h.outputColumn]
  simp only [active,offset,accumulation_eq P p h]
  rfl

end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteSource
