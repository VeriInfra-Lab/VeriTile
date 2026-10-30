import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafetyTools

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


theorem prefix_steps :
    List.Forall₂ (fun a b => ∀ s, Safety.stepSafe (fusedAlgebra native g) bounds a s ↔
      Safety.stepSafe (WholeNative.algebra native g) bounds b s)
      (SourceCode.candidate N M BN BM rowShape columnShape).before
      (SourceCode.original N M BN BM rowShape columnShape).before := by
  simp only [SourceCode.candidate,SourceCode.original,Source.whileLoop?,Kernels.source,
    Kernels.explicitFMA,ComputeKernel.surfaceBody,Option.get_some,List.forall₂_cons]
  repeat' constructor
  all_goals
    intro s
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalExpr,evalComputeOp,evalOp_unfold,
      fusedAlgebra,WholeNative.algebra,numeric,bop,ofFloat,toFloat,
      ComputeDType.eraseDType,Option.bind,store,Broadcast.leftIndex,Broadcast.rightIndex]


theorem prefix_equal (s : State Payload) :
    Safety.traceSafeList (fusedAlgebra native g) bounds
      (SourceCode.candidate N M BN BM rowShape columnShape).before s ↔
    Safety.traceSafeList (WholeNative.algebra native g) bounds
      (SourceCode.original N M BN BM rowShape columnShape).before s :=
  trace_congr_steps _ _ bounds (SourcePair.prefix_steps native N M BN BM g rowShape columnShape)
    (prefix_steps native N M BN BM g rowShape columnShape bounds) s


theorem body_steps :
    List.Forall₂ (fun a b => ∀ s, Safety.stepSafe (fusedAlgebra native g) bounds a s ↔
      Safety.stepSafe (WholeNative.algebra native g) bounds b s)
      (SourceCode.candidate N M BN BM rowShape columnShape).body
      (SourceCode.original N M BN BM rowShape columnShape).body := by
  simp only [SourceCode.candidate,SourceCode.original,Source.whileLoop?,Kernels.source,
    Kernels.explicitFMA,ComputeKernel.surfaceBody,Option.get_some,List.forall₂_cons]
  repeat' constructor
  all_goals
    intro s
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalExpr,evalComputeOp,evalOp_unfold,
      fusedAlgebra,WholeNative.algebra,numeric,bop,ofFloat,toFloat,
      ComputeDType.eraseDType,Option.bind,store,Broadcast.leftIndex,Broadcast.rightIndex]


theorem body_equal (s : State Payload) :
    Safety.traceSafeList (fusedAlgebra native g) bounds
      (SourceCode.candidate N M BN BM rowShape columnShape).body s ↔
    Safety.traceSafeList (WholeNative.algebra native g) bounds
      (SourceCode.original N M BN BM rowShape columnShape).body s :=
  trace_congr_steps _ _ bounds (SourcePair.body_steps native N M BN BM g rowShape columnShape)
    (body_steps native N M BN BM g rowShape columnShape bounds) s


theorem suffix_steps :
    List.Forall₂ (fun a b => ∀ s, Safety.stepSafe (fusedAlgebra native g) bounds a s ↔
      Safety.stepSafe (WholeNative.algebra native g) bounds b s)
      (SourceCode.candidate N M BN BM rowShape columnShape).after
      (SourceCode.original N M BN BM rowShape columnShape).after := by
  simp only [SourceCode.candidate,SourceCode.original,Source.whileLoop?,Kernels.source,
    Kernels.explicitFMA,ComputeKernel.surfaceBody,Option.get_some,List.forall₂_cons]
  repeat' constructor
  all_goals
    intro s
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalExpr,evalComputeOp,evalOp_unfold,
      fusedAlgebra,WholeNative.algebra,numeric,bop,ofFloat,toFloat,
      ComputeDType.eraseDType,Option.bind,store,Broadcast.leftIndex,Broadcast.rightIndex]


theorem suffix_equal (s : State Payload) :
    Safety.traceSafeList (fusedAlgebra native g) bounds
      (SourceCode.candidate N M BN BM rowShape columnShape).after s ↔
    Safety.traceSafeList (WholeNative.algebra native g) bounds
      (SourceCode.original N M BN BM rowShape columnShape).after s :=
  trace_congr_steps _ _ bounds (SourcePair.suffix_steps native N M BN BM g rowShape columnShape)
    (suffix_steps native N M BN BM g rowShape columnShape bounds) s


end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePairSafety
