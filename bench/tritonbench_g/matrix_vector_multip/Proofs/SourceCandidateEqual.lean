import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateFull
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateSafety
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

theorem range_equal (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (initial current : State Payload) (off : Nat)
    (acc : WholeSource.Acc Payload p) (h : RuntimeInvariant p initial off acc current) :
    range (model native p g) "m" off p.columns p.blockColumns (loopBody p) current=
      range (WholeNative.algebra native g) "m" off p.columns p.blockColumns
        (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.loopBody p) current := by
  by_cases active : off<p.columns
  · rw [range,if_neg (by omega),if_pos active]
    conv_rhs => rw [range,if_neg (by omega),if_pos active]
    rw [show current.setReg "m" .nat [] (fun _ => off)=current from h.counter,
      (bodyExecution native p g initial current off acc h).property,
      (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.bodyExecution (WholeNative.algebra native g) p
        (by intros; rfl) (by intros; rfl) initial current off acc h).property]
    exact range_equal native p g width initial _ (off+p.blockColumns) _
      (bodyInvariant (WholeNative.algebra native g) p initial current off acc h)
  · rw [range,if_neg (by omega),if_neg active]
    conv_rhs => rw [range,if_neg (by omega),if_neg active]
termination_by p.columns-off

theorem execution_equal (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (initial : State Payload)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : initial.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64) :
    exec (model native p g) (source p) initial=
      exec (WholeNative.algebra native g) (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.source p) initial := by
  let A := WholeNative.algebra native g
  let start := VeriTile.Bench.Repairs.MatrixVectorMultip.Source.preludeExecution A p initial
  have inv := preludeInvariant A p initial small pidSmall rowsSmall colsSmall
  change run _ (source p).surfaceBody initial=run _ (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.source p).surfaceBody initial
  rw [source_decomposition,VeriTile.Bench.Repairs.MatrixVectorMultip.Source.source_decomposition,
    run_append,run_append,prelude_run,start.property]
  simp only [Option.bind_some,run]
  rw [dynamicLoop native p g initial start.val 0 _ inv,
    VeriTile.Bench.Repairs.MatrixVectorMultip.Source.dynamicLoop A p initial start.val 0 _ inv,
    range_equal native p g width initial start.val 0 _ inv]
  congr 1
  funext state
  exact epilogue_run native p g state
#axiomsClean range_equal
#axiomsClean execution_equal
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
