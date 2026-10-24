import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelAlgebra
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.KernelNativeResults
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native (Term Word Instructions Schedule Reductions)
noncomputable section

/-- The original and integer-commuted source execute to the same complete raw
state under the unchanged native primitive model, preserving the full frame. -/
theorem softmax_row_fp32 (R : Reductions) (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder)
    (Y X : RegionName) (M N : Nat) (positive : 0 < N) (before : State Term) :
    ∃ after, FP.Structural.exec (FP.ProfiledRewrite.engine (KernelNative.algebra R I S O))
        (kernel Y X M N (Config.tile N)) before=some after ∧
      FP.Structural.exec (FP.ProfiledRewrite.engine (KernelNative.algebra R I S O))
        (indexCommutedKernel Y X M N (Config.tile N)) before=some after ∧
      FPExecution.RowFrame Y N (before.pids 0) before after := by
  obtain ⟨after,run,frame⟩ := FPExecution.complete_row
    (VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.algebra R) (KernelNative.equal I S O)
    Y X M N (Config.tile N) positive (Config.tile_positive _) before
  exact ⟨after,run,(KernelRewrite.execution (KernelNative.algebra R I S O)
    Y X M N (Config.tile N) before).symm.trans run,frame⟩

#axiomsClean softmax_row_fp32
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.KernelNativeResults
