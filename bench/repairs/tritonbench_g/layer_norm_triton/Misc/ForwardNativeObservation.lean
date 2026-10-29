import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardEquivalenceIO
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.NativeObservationDefs

/-! Auxiliary native observation of the forward stage. Primary stage results are in FPEquiv. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Triton FP.Structural SourceModel SourceNative SourceNative.Native Interface
noncomputable section

/-- Independent ordered forward arithmetic with native word observation. -/
theorem forward_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (stride N B row : Nat) (eps : ℝ) (positive : 0<N) (covers : N≤B) :
    (Forward.io (model R) stride N B row eps (observeWords I S)).Implements
      (ForwardIO.expected (model R) stride N B row eps (observeWords I S)) :=
  ForwardIO.implements (model R) stride N B row eps (observeWords I S) positive covers

#specNonCircular ForwardIO.expected avoiding [FP.Structural.exec,FP.Structural.run,TypedSource.forward]

#stmtSurfaceSubset forward_native ⊆ [Term,Reductions,Instructions,OutputSchedule,Forward.io,model,observeWords,
  ProgramIO.Implements,ForwardIO.expected,FastLayernorm.KernelNative.instInhabitedTerm]

#axiomsClean forward_native

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
