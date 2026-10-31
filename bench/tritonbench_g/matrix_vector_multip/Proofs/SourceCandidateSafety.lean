import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateSafetySteps
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem range_safe (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (bounds : RegionName → Nat)
    (initial current : State Payload) (off : Nat) (acc : WholeSource.Acc Payload p)
    (h : RuntimeInvariant p initial off acc current)
    (matrix : ∀ off, ∀ i : Fin p.blockRows, ∀ j : Fin p.blockColumns,
      WholeSource.row p (initial.pids 0) i.val<p.rows → off+j.val<p.columns →
        WholeSource.aOffset p (initial.pids 0) off i.val j.val<bounds p.a)
    (vector : ∀ off, ∀ j : Fin p.blockColumns, off+j.val<p.columns → WholeSource.bOffset p off j.val<bounds p.b) :
    Safety.rangeSafe (model native p g) bounds "m" off p.columns p.blockColumns (loopBody p) current := by
  by_cases active : off<p.columns
  · rw [Safety.rangeSafe_unfold]
    simp only [Nat.ne_of_gt width,if_false,active,if_true]
    refine ⟨?_,?_⟩
    · rw [show current.setReg "m" .nat [] (fun _ => off)=current from h.counter]
      exact body_safe native p g bounds initial current off acc h (matrix off) (vector off)
    · rw [show current.setReg "m" .nat [] (fun _ => off)=current from h.counter,
        (bodyExecution native p g initial current off acc h).property]
      exact range_safe native p g width bounds initial _ (off+p.blockColumns) _
        (bodyInvariant (WholeNative.algebra native g) p initial current off acc h) matrix vector
  · rw [Safety.rangeSafe_unfold]
    simp [Nat.ne_of_gt width,active]
termination_by p.columns-off

theorem loop_safe (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (bounds : RegionName → Nat)
    (initial current : State Payload) (off : Nat) (acc : WholeSource.Acc Payload p)
    (h : RuntimeInvariant p initial off acc current)
    (matrix : ∀ off, ∀ i : Fin p.blockRows, ∀ j : Fin p.blockColumns,
      WholeSource.row p (initial.pids 0) i.val<p.rows → off+j.val<p.columns →
        WholeSource.aOffset p (initial.pids 0) off i.val j.val<bounds p.a)
    (vector : ∀ off, ∀ j : Fin p.blockColumns, off+j.val<p.columns → WholeSource.bOffset p off j.val<bounds p.b) :
    Safety.stepSafe (model native p g) bounds (loopStatement p) current := by
  have bound := h.metadata "M" p.columns (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  rw [loopStatement_eq]
  simp [Safety.stepSafe_unfold,Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeExpr_unfold,
    evalOp_unfold,evalExpr,evalComputeOp,Option.bind,bound,counter]
  exact range_safe native p g width bounds initial current off acc h matrix vector

theorem source_safe (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (bounds : RegionName → Nat) (initial : State Payload)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : initial.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64)
    (matrix : ∀ i j, i<p.rows → j<p.columns → i*p.strideRow+j*p.strideColumn<bounds p.a)
    (vector : ∀ j, j<p.columns → j*p.strideVector<bounds p.b)
    (output : ∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
      WholeSource.row p (initial.pids 0) i.val*p.strideOut<bounds p.out) :
    TraceSafe (model native p g) bounds (source p) initial := by
  have hm : ∀ off, ∀ i : Fin p.blockRows, ∀ j : Fin p.blockColumns,
      WholeSource.row p (initial.pids 0) i.val<p.rows → off+j.val<p.columns →
        WholeSource.aOffset p (initial.pids 0) off i.val j.val<bounds p.a := by
    intro off i j activeRow activeColumn
    convert matrix _ (off+j.val) activeRow activeColumn using 1
    simp [WholeSource.aOffset,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  have hv : ∀ off, ∀ j : Fin p.blockColumns, off+j.val<p.columns →
      WholeSource.bOffset p off j.val<bounds p.b := by
    intro off j active
    simpa [WholeSource.bOffset,Nat.add_mul,Nat.add_comm] using vector (off+j.val) active
  let A := WholeNative.algebra native g
  let start := VeriTile.Bench.Repairs.MatrixVectorMultip.Source.preludeExecution A p initial
  have inv := preludeInvariant A p initial small pidSmall rowsSmall colsSmall
  unfold TraceSafe
  rw [source_decomposition,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [prelude_eq]
    exact VeriTile.Bench.Repairs.MatrixVectorMultip.Source.prelude_safe (model native p g) p bounds initial
  intro after ran
  have equal : after=start.val := Option.some.inj (ran.symm.trans ((prelude_run native p g initial).trans start.property))
  subst after
  obtain ⟨last,middle,loop,finalInv⟩ := rangeExecution native p g width initial start.val 0 _ inv
  rw [Safety.traceSafeList_unfold]
  refine ⟨loop_safe native p g width bounds initial start.val 0 _ inv hm hv,?_⟩
  rw [dynamicLoop native p g initial start.val 0 _ inv,loop,epilogue_eq]
  exact VeriTile.Bench.Repairs.MatrixVectorMultip.Source.epilogue_safe (model native p g) p bounds initial middle last _ finalInv
    output

#axiomsClean range_safe
#axiomsClean loop_safe
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
