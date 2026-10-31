import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceBodyInvariant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loopStatement_eq (p : Parameters) :
    loopStatement p=.forRangeDyn "m" (.ref .nat [] "m") (.ref .nat [] "M")
      (.constNat p.blockColumns) (loopBody p) := by rfl

theorem dynamicLoop (A : Algebra α) (p : Parameters) (initial current : State α)
    (off : Nat) (acc : WholeSource.Acc α p) (h : RuntimeInvariant p initial off acc current) :
    step A (loopStatement p) current=range A "m" off p.columns p.blockColumns (loopBody p) current := by
  have bound := h.metadata "M" p.columns (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  rw [loopStatement_eq]
  simp [step,evalOp_unfold,bound,counter]

theorem rangeExecution (A : Algebra α) (p : Parameters) (width : 0<p.blockColumns)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current) :
    ∃ last final, range A "m" off p.columns p.blockColumns (loopBody p) current=some final ∧
      RuntimeInvariant p initial last
        (WholeSource.accumulated A p (initial.pids 0) (viewMemory initial.mem) off acc) final := by
  by_cases active : off<p.columns
  · let next := bodyExecution A p castId loadId initial current off acc h
    obtain ⟨last,final,ran,inv⟩ := rangeExecution A p width castId loadId initial next.val (off+p.blockColumns)
      (WholeSource.nextAcc A p (initial.pids 0) off (viewMemory initial.mem) acc)
      (bodyInvariant A p initial current off acc h)
    refine ⟨last,final,?_,?_⟩
    · rw [range,if_neg (by omega),if_pos active,
        show current.setReg "m" .nat [] (fun _ => off)=current from h.counter,next.property]
      exact ran
    · rw [WholeSource.accumulated]
      simpa only [active,width,and_self,if_true] using inv
  · refine ⟨off,current,?_,?_⟩
    · rw [range,if_neg (by omega),if_neg active]
    · rw [WholeSource.accumulated]
      simpa only [active,false_and,if_false] using h
termination_by p.columns-off

#axiomsClean loopStatement_eq
#axiomsClean dynamicLoop
#axiomsClean rangeExecution
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
