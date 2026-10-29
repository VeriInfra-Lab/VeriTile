import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafetySteps

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePairSafety
open VeriTile.Triton FP.Structural SourcePair
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

variable (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
  (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
  (bounds : RegionName → Nat)


theorem loop_equal (s : State Payload) :
    Safety.stepSafe (fusedAlgebra native g) bounds
      (.whileLoop (SourceCode.candidate N M BN BM rowShape columnShape).condition
        (SourceCode.candidate N M BN BM rowShape columnShape).body) s ↔
    Safety.stepSafe (WholeNative.algebra native g) bounds
      (.whileLoop (SourceCode.original N M BN BM rowShape columnShape).condition
        (SourceCode.original N M BN BM rowShape columnShape).body) s := by
  rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
  have body := funext (SourcePair.body_equal native N M BN BM g rowShape columnShape)
  have guardLeft (current : State Payload) :
      Safety.safeOp (fusedAlgebra native g) none bounds current
        (SourceCode.candidate N M BN BM rowShape columnShape).condition := by
    simp [SourceCode.candidate,Source.whileLoop?,Kernels.explicitFMA,ComputeKernel.surfaceBody,
      Safety.safeOp_unfold]
  have guardRight (current : State Payload) :
      Safety.safeOp (WholeNative.algebra native g) none bounds current
        (SourceCode.original N M BN BM rowShape columnShape).condition := by
    simp [SourceCode.original,Source.whileLoop?,Kernels.source,ComputeKernel.surfaceBody,
      Safety.safeOp_unfold]
  constructor
  · intro safe current reached
    have originalReach : Control.While.Reaches
        (fun state => (evalOp (fusedAlgebra native g) none
          (SourceCode.candidate N M BN BM rowShape columnShape).condition state).map
          (fun c => c PUnit.unit))
        (run (fusedAlgebra native g) (SourceCode.candidate N M BN BM rowShape columnShape).body)
        s current := by
      rw [body]
      exact reached
    exact ⟨guardRight current,fun active =>
      (body_equal native N M BN BM g rowShape columnShape bounds current).mp
        ((safe current originalReach).2 active)⟩
  · intro safe current reached
    have originalReach : Control.While.Reaches
        (fun state => (evalOp (WholeNative.algebra native g) none
          (SourceCode.original N M BN BM rowShape columnShape).condition state).map
          (fun c => c PUnit.unit))
        (run (WholeNative.algebra native g) (SourceCode.original N M BN BM rowShape columnShape).body)
        s current := by
      rw [body] at reached
      exact reached
    exact ⟨guardLeft current,fun active =>
      (body_equal native N M BN BM g rowShape columnShape bounds current).mpr
        ((safe current originalReach).2 active)⟩


theorem source_equal (s : State Payload) :
    TraceSafe (fusedAlgebra native g) bounds (Kernels.explicitFMA N M BN BM rowShape columnShape) s ↔
    TraceSafe (WholeNative.algebra native g) bounds (Kernels.source N M BN BM rowShape columnShape) s := by
  unfold TraceSafe
  rw [SourceCode.candidate_decomposition,SourceCode.original_decomposition]
  apply trace_congr_steps
  · exact append_steps
      (append_steps (SourcePair.prefix_steps native N M BN BM g rowShape columnShape)
        (List.Forall₂.cons (SourcePair.loop_equal native N M BN BM g rowShape columnShape) .nil))
      (SourcePair.suffix_steps native N M BN BM g rowShape columnShape)
  · exact append_steps
      (append_steps (prefix_steps native N M BN BM g rowShape columnShape bounds)
        (List.Forall₂.cons (loop_equal native N M BN BM g rowShape columnShape bounds) .nil))
      (suffix_steps native N M BN BM g rowShape columnShape bounds)


end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePairSafety
