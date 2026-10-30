import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.RawKernel
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Gradient

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete raw forward/backward execution and the independent directional
derivative, with every output and both phase frames. -/
theorem kernel_correct (input : Input ℝ) (BH T D BDF BDB : Nat)
    (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) (initial : State ℝ)
    (reads : Execution.Inputs initial input T D) :
    ∃ f b, RawKernel.Result realOps input BH T D BDF BDB U S V initial f b ∧
      (∀ bh d, bh<BH → d<D → ∀ vx vg : Nat → ℝ,
        HasDerivAt (fun e =>
          (∑ time∈Finset.range T,input "do" bh time d*pathState input U bh d vx vg e (time+1))+
          terminal realOps V input bh d*pathState input U bh d vx vg e T)
          (∑ time∈Finset.range T,
            (Execution.read b "dx" (rowAddr bh T D time d)*vx time+
             Execution.read b "dg" (rowAddr bh T D time d)*vg time)) 0) := by
  obtain ⟨f,b,h⟩ := RawKernel.complete realOps input BH T D BDF BDB hF hB U S V initial reads
  refine ⟨f,b,h,?_⟩
  intro bh d hh hd vx vg
  have equal : (∑ time∈Finset.range T,
      (Execution.read b "dx" (rowAddr bh T D time d)*vx time+
       Execution.read b "dg" (rowAddr bh T D time d)*vg time))=
    ∑ time∈Finset.range T,(dx realOps V input T bh d time*vx time+
      dg realOps U V input T bh d time*vg time) := by
    apply Finset.sum_congr rfl
    intro time ht
    simp only [Execution.read,(h.gradients bh hh time (Finset.mem_range.mp ht) d hd).1,
      (h.gradients bh hh time (Finset.mem_range.mp ht) d hd).2,Cell.read_same]
  rw [equal]
  exact loss_derivative input U V T bh d vx vg

#axiomsClean kernel_correct
#specNonCircular pathState avoiding [Execution.gridRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.KernelRealResult
