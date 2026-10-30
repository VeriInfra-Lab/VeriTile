import bench.pretrain_kernels.tasks.local_mtp_tail_mask_open_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mtp_tail_mask_open_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  (∀ (j0 : Fin 6), xs.constantcb21bbcb16 j0 = ([0, 16, 48, 80, 128, 128] : List Int).getD (j0.val) 0)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let input0 := xs.input0
    let constantcb21bbcb16 := xs.constantcb21bbcb16
    let vfill : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 15) then vfill i0 else 1)
    let vfill_1 : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter_1 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 47) then vfill_1 i0 else vselect_scatter i0 i1)
    let vfill_2 : Fin 2 → ℝ := fun i0 => (0 : ℝ)
    let vselect_scatter_2 : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val = 79) then vfill_2 i0 else vselect_scatter_1 i0 i1)
    let vmul : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (input0 i0 i1 * vselect_scatter_2 i0 i1)
    vmul i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mtp_tail_mask_open_fwd
