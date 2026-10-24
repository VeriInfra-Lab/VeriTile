import bench.repairs.tritonbench_g.fast_rms_layernorm.Misc.NativeReference
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativePublicIO

/-! Retained single-source native correctness. These ordinary theorems keep
the original names, full contracts and proofs; they do not assert an
implementation-pair floating-point equivalence. -/

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv
open VeriTile Triton FP.Structural NativeGraph
noncomputable section

/-- Native observation of the complete forward source with its actual runtime
FP32 epsilon ABI. Primitive instruction semantics and topology/fusion choices
remain model parameters; no numerical transformation assumption is added. -/
theorem kernel_forward (R : Reductions) (S : Selection) (I : Instructions)
    (h : Call) (epsilon : EpsilonABI.Binding h.eps) :
    (NativeIO.program R h Bool.false (.epsilon epsilon.runtime)).Observes (Reference.observer h Bool.false I S)
      (Reference.nativeWords R S I h Bool.false (.epsilon (EpsilonABI.convert epsilon.host))) := by
  have observed:=ProgramIO.Observes.of_implements
    (NativePublicIO.forward R h (.epsilon epsilon.runtime)) (fun _ _=>eval I S)
  simpa only [Reference.nativeWords,ProgramIO.mapOutputs,epsilon.converted] using observed

/-- Both complete backward grids, all arbitrary symbolic inputs, all FP32
output words and the full raw-cell frame. The zero-DW domain applies only to
empty rows and comes from the existing explicit host initialization. -/
theorem kernel_backward (R : Reductions) (S : Selection) (I : Instructions) (h : Call) :
    (NativeIO.program R h Bool.true (.bits 0)).ObservesOn (Reference.observer h Bool.true I S)
      (Reference.nativeDomain h Bool.true) (Reference.nativeWords R S I h Bool.true (.bits 0)) := by
  exact ProgramIO.ObservesOn.of_implementsOn (NativePublicIO.backward R h (.bits 0)) (fun _ _=>eval I S)

#specNonCircular Reference.nativeWords avoiding [Kernels.forwardRuntime,Kernels.forward,Kernels.backward,Kernels.weightBackward,FP.Structural.exec]
#specNonCircular Reference.nativeDomain avoiding [Kernels.forwardRuntime,Kernels.forward,Kernels.backward,Kernels.weightBackward,FP.Structural.exec]
#stmtSurfaceSubset kernel_forward ⊆
  [Call,Call.eps,Reductions,Selection,Instructions,NativeIO.program,Reference.nativeWords,
   ProgramIO.Observes,Reference.observer,Term,Term.epsilon,Word,NativeGraph.instInhabitedTerm,NativeGraph.eval,
   EpsilonABI.Binding,EpsilonABI.Binding.runtime,EpsilonABI.Binding.host,EpsilonABI.convert]
#stmtSurfaceSubset kernel_backward ⊆
  [Call,Reductions,Selection,Instructions,NativeIO.program,Reference.nativeWords,Reference.nativeDomain,
   ProgramIO.ObservesOn,Reference.observer,Term,Term.bits,Word,NativeGraph.instInhabitedTerm,NativeGraph.eval]
#axiomsClean kernel_forward
#axiomsClean kernel_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv
