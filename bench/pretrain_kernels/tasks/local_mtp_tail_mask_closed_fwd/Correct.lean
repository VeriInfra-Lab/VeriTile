import bench.pretrain_kernels.tasks.local_mtp_tail_mask_closed_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_mtp_tail_mask_closed_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed (fun xs (i0,i1) =>
    let input0 := xs.input0
    let constant53d3e3c65e := xs.constant53d3e3c65e
    let vfill : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 15) then vfill i0 else 1)
    let vfill_1 : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter_1 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 47) then vfill_1 i0 else vselect_scatter i0 i1)
    let vfill_2 : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter_2 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 79) then vfill_2 i0 else vselect_scatter_1 i0 i1)
    let vfill_3 : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter_3 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 127) then vfill_3 i0 else vselect_scatter_2 i0 i1)
    let vmul : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (input0 i0 i1 * vselect_scatter_3 i0 i1)
    vmul i0 i1))

end
end VeriTile.Bench.PretrainKernels.local_mtp_tail_mask_closed_fwd
