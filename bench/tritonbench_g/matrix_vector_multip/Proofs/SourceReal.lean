import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceReadback
import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_WholeMath
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Full logical column sum from initial matrix/vector memory only. -/
def realTarget (p : Parameters) (memory : RegionName → Nat → Cell ℝ) (row : Nat) : ℝ :=
  ∑ j : Fin p.columns,
    (memory p.a (row*p.strideRow+j.val*p.strideColumn)).read .real *
      (memory p.b (j.val*p.strideVector)).read .real

theorem real_value (p : Parameters) (width : 0<p.blockColumns) (initial : State ℝ)
    (i : Fin p.blockRows) (active : WholeSource.row p (initial.pids 0) i.val<p.rows) :
    WholeSource.tileValue WholeMath.algebra p (initial.pids 0) (viewMemory initial.mem) i =
      realTarget p initial.mem (WholeSource.row p (initial.pids 0) i.val) := by
  rw [WholeMath.tile_mathematical p width _ _ i active]
  apply Finset.sum_congr rfl
  intro j _
  simp [WholeMath.product,realTarget,viewMemory,storageView_read]

theorem real_correct (p : Parameters) (width : 0<p.blockColumns) (initial : State ℝ)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : initial.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64)
    (stride : p.strideOut=1) :
    ∃ final, exec WholeMath.algebra (source p) initial=some final ∧
      (∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
        final.mem p.out (WholeSource.row p (initial.pids 0) i.val)=Cell.mk .real
          (realTarget p initial.mem (WholeSource.row p (initial.pids 0) i.val))) ∧
      (∀ r o, (r≠p.out ∨ ∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
        o≠WholeSource.row p (initial.pids 0) i.val) → final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,read,frame⟩ := source_correct WholeMath.algebra p width (by intros; rfl)
    (by intros; rfl) initial small pidSmall rowsSmall colsSmall stride
  refine ⟨final,ran,?_,frame⟩
  intro i active
  rw [read i active,real_value p width initial i active]

#specNonCircular realTarget avoiding [source,FP.Structural.exec,WholeSource.tileValue]
#axiomsClean real_value
#axiomsClean real_correct
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
