import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceBodyExecution
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem bodyInvariant (A : Algebra α) (p : Parameters) (initial current : State α)
    (off : Nat) (acc : WholeSource.Acc α p) (h : RuntimeInvariant p initial off acc current) :
    RuntimeInvariant p initial (off+p.blockColumns)
      (WholeSource.nextAcc A p (initial.pids 0) off (viewMemory initial.mem) acc)
      (bodyResult A p initial current off acc) := by
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  refine ⟨?_,?_,?_,?_⟩
  · simp [WholeSource.Context,bodyResult,hoffn,hoffm,hnMask,hmem]
  · simpa [bodyResult] using h.pids
  · intro name n member
    have unchanged : (bodyResult A p initial current off acc).regs .nat [] name = current.regs .nat [] name := by
      simp only [scalarParameters,List.mem_cons,List.not_mem_nil,or_false] at member
      rcases member with he | he | he | he | he | he
      all_goals cases he
      all_goals simp [bodyResult]
    rw [unchanged]
    exact h.metadata name n member
  · exact AffineWhile.normalized_set _ _ _

#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
