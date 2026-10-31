import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceLoopExecution
import VeriTile.Triton.Float.AffineWhileInvariant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem whileExecution (A : Algebra α) (p : Parameters) (width : 0<p.blockColumns)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current) :
    ∃ last final, range A "m" off p.columns p.blockColumns (loopBody p) current=some final ∧
      AffineWhile.Trace A (AffineWhile.guard "m" "M") (loopBody p) current final ∧
      RuntimeInvariant p initial last
        (WholeSource.accumulated A p (initial.pids 0) (viewMemory initial.mem) off acc) final := by
  obtain ⟨last,final,ran,result⟩ := rangeExecution A p width castId loadId initial current off acc h
  refine ⟨last,final,ran,?_,result⟩
  let I (n : Nat) (state : State α) : Prop := ∃ value, RuntimeInvariant p initial n value state
  apply AffineWhile.range_to_trace_invariant A "m" "M" p.blockColumns p.columns width (loopBody p) I
  · intro n state inv
    obtain ⟨value,inv⟩ := inv
    exact ⟨inv.counter,inv.metadata "M" p.columns (by simp [scalarParameters])⟩
  · intro n state after inv executed
    obtain ⟨value,inv⟩ := inv
    let actual := bodyExecution A p castId loadId initial state n value inv
    have equal : after=actual.val := Option.some.inj (executed.symm.trans actual.property)
    subst after
    exact ⟨_,bodyInvariant A p initial state n value inv⟩
  · exact ⟨acc,h⟩
  · exact ran

#axiomsClean whileExecution
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
