import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceFullExecution
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem written_value (A : Algebra α) (p : Parameters) (initial : State α)
    (acc : WholeSource.Acc α p) (stride : p.strideOut=1)
    (i : Fin p.blockRows) (active : WholeSource.row p (initial.pids 0) i.val<p.rows) :
    (written A p initial acc).mem p.out (WholeSource.row p (initial.pids 0) i.val)=
      Cell.mk .real (WholeSource.outputValues A p acc (i,())) := by
  have result := State.masked_scatter_readback p.out
    (fun i : TileIndex [p.blockRows,1] => WholeSource.row p (initial.pids 0) i.1.val*p.strideOut)
    (fun i => Cell.mk .real (WholeSource.outputValues A p acc (i.1,())))
    (fun i => WholeSource.row p (initial.pids 0) i.1.val<p.rows)
    initial (i,0,()) active (WholeSource.row_injective p (initial.pids 0) stride)
  simpa only [written,scatter,stride,Nat.mul_one] using result

theorem written_frame (A : Algebra α) (p : Parameters) (initial : State α)
    (acc : WholeSource.Acc α p) (stride : p.strideOut=1) (r : RegionName) (o : Nat)
    (outside : r≠p.out ∨ ∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
      o≠WholeSource.row p (initial.pids 0) i.val) :
    (written A p initial acc).mem r o=initial.mem r o := by
  apply State.masked_scatter_frame
  rcases outside with region | address
  · exact Or.inl region
  · right
    intro i _ active
    simpa only [stride,Nat.mul_one] using address i.1 active

theorem source_correct (A : Algebra α) (p : Parameters) (width : 0<p.blockColumns)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value) (initial : State α)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : initial.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64)
    (stride : p.strideOut=1) :
    ∃ final, exec A (source p) initial=some final ∧
      (∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
        final.mem p.out (WholeSource.row p (initial.pids 0) i.val)=Cell.mk .real
          (WholeSource.tileValue A p (initial.pids 0) (viewMemory initial.mem) i)) ∧
      (∀ r o, (r≠p.out ∨ ∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
        o≠WholeSource.row p (initial.pids 0) i.val) → final.mem r o=initial.mem r o) := by
  obtain ⟨middle,final,ran,trace,memory⟩ := source_execution_memory A p width castId loadId initial small pidSmall rowsSmall colsSmall
  refine ⟨final,ran,?_,?_⟩
  · intro i active
    rw [memory,written_value A p initial _ stride i active]
    simp only [WholeSource.tileValue,WholeSource.storedValue,castId]
  · intro r o outside
    rw [memory]
    exact written_frame A p initial _ stride r o outside

#axiomsClean written_value
#axiomsClean written_frame
#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
