import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateExecution
import VeriTile.Triton.Float.AffineWhileInvariant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

theorem whileExecution (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (initial current : State Payload) (off : Nat)
    (acc : WholeSource.Acc Payload p) (h : RuntimeInvariant p initial off acc current) :
    ∃ last final, range (model native p g) "m" off p.columns p.blockColumns (loopBody p) current=some final ∧
      AffineWhile.Trace (model native p g) (AffineWhile.guard "m" "M") (loopBody p) current final ∧
      RuntimeInvariant p initial last (WholeSource.accumulated (WholeNative.algebra native g) p
        (initial.pids 0) (viewMemory initial.mem) off acc) final := by
  obtain ⟨last,final,ran,result⟩ := rangeExecution native p g width initial current off acc h
  refine ⟨last,final,ran,?_,result⟩
  let I (n : Nat) (state : State Payload) : Prop := ∃ value, RuntimeInvariant p initial n value state
  apply AffineWhile.range_to_trace_invariant (model native p g) "m" "M" p.blockColumns p.columns width (loopBody p) I
  · intro n state inv
    obtain ⟨value,inv⟩ := inv
    exact ⟨inv.counter,inv.metadata "M" p.columns (by simp [scalarParameters])⟩
  · intro n state after inv executed
    obtain ⟨value,inv⟩ := inv
    let actual := bodyExecution native p g initial state n value inv
    have equal : after=actual.val := Option.some.inj (executed.symm.trans actual.property)
    subst after
    exact ⟨_,bodyInvariant (WholeNative.algebra native g) p initial state n value inv⟩
  · exact ⟨acc,h⟩
  · exact ran

theorem source_execution_memory (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (initial : State Payload)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : initial.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64) :
    ∃ middle final, exec (model native p g) (source p) initial=some final ∧
      AffineWhile.Trace (model native p g) (AffineWhile.guard "m" "M") (loopBody p)
        (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.preludeExecution (WholeNative.algebra native g) p initial).val middle ∧
      final.mem=(written (WholeNative.algebra native g) p initial
        (WholeSource.accumulated (WholeNative.algebra native g) p (initial.pids 0)
          (viewMemory initial.mem) 0 (fun _ => .word 0))).mem := by
  let A := WholeNative.algebra native g
  let start := VeriTile.Bench.Repairs.MatrixVectorMultip.Source.preludeExecution A p initial
  have startReady := preludeInvariant A p initial small pidSmall rowsSmall colsSmall
  obtain ⟨last,middle,loop,trace,inv⟩ := whileExecution native p g width initial start.val 0
    (fun _ => .word 0) startReady
  let finish := epilogueExecution A p (by intros; rfl) initial middle last _ inv
  refine ⟨middle,finish.val,?_,trace,epilogueMemory A p (by intros; rfl) initial middle last _ inv⟩
  change run (model native p g) (source p).surfaceBody initial=some finish.val
  rw [source_decomposition,run_append,prelude_run,start.property]
  simp only [Option.bind_some,run,dynamicLoop native p g initial start.val 0 _ startReady,loop]
  change run (model native p g) (epilogue p) middle=some finish.val
  rw [epilogue_run]
  exact finish.property

#axiomsClean whileExecution
#axiomsClean source_execution_memory
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
