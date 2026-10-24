import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.RawKernel
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Scalar

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.KernelNativeResult
open VeriTile.Triton FP.Structural
noncomputable section

/-- Raw FP32 source execution and native output observations under the original
three local scalar laws and value-independent compiler schedule. -/
theorem kernel_fp32 (p : Native.Instructions) (laws : Native.LocalConformance p)
    (schedule : Native.Schedule) (input : Input Native.Term) (BH T D BDF BDB : Nat)
    (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) (initial : State Native.Term)
    (reads : Execution.Inputs initial input T D) :
    ∃ f b, RawKernel.Result Native.sourceOps input BH T D BDF BDB U S V initial f b ∧
      (∀ bh time d, bh<BH → time<T → d<D →
        Native.observe p schedule b "o" (rowAddr bh T D time d)=
          Native.eval Native.software schedule (state Native.sourceOps U input bh d (time+1)) ∧
        Native.observe p schedule b "dx" (rowAddr bh T D time d)=
          Native.eval Native.software schedule (dx Native.sourceOps V input T bh d time) ∧
        Native.observe p schedule b "dg" (rowAddr bh T D time d)=
          Native.eval Native.software schedule (dg Native.sourceOps U V input T bh d time)) ∧
      (S=Bool.true → ∀ bh d, bh<BH → d<D →
        Native.observe p schedule b "ht" (stateAddr bh D d)=
          Native.eval Native.software schedule (state Native.sourceOps U input bh d T)) := by
  obtain ⟨f,b,h⟩ := RawKernel.complete Native.sourceOps input BH T D BDF BDB hF hB U S V initial reads
  refine ⟨f,b,h,?_,?_⟩
  · intro bh time d hh ht hd
    simp only [Native.observe,h.output bh hh time ht d hd,
      (h.gradients bh hh time ht d hd).1,(h.gradients bh hh time ht d hd).2,
      Cell.read_same,Native.software_eq p laws,true_and]
  · intro hs bh d hh hd
    simp only [Native.observe,h.finalState hs bh hh d hd,Cell.read_same,Native.software_eq p laws]

#axiomsClean kernel_fp32
#specNonCircular Native.LocalConformance avoiding [Execution.gridRun,FP.Structural.exec,state,dx,dg,Native.eval]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.KernelNativeResult
