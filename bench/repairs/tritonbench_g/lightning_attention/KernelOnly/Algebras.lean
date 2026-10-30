import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.MaskArithmetic
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.RealInputs
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.NativeTerms
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
/-- Real execution of the source's previously omitted floating comparisons. -/
abbrev realAlgebra := Real.algebra

def realMasks : MaskArithmetic realAlgebra where
  le := fun a b => decide (a ≤ b)
  compare := rfl
  lower := by
    intro i j _ _
    change decide ((0 : ℝ) ≤ (i : ℝ) - (j : ℝ)) = decide (j ≤ i)
    congr 1
    simp
  upper := by
    intro i j _ _
    change decide ((i : ℝ) - (j : ℝ) ≤ (0 : ℝ)) = decide (i ≤ j)
    congr 1
    simp

abbrev nativeAlgebra := Native.algebra

def nativeMasks : MaskArithmetic nativeAlgebra where
  le := fun a b => !FP.lt .ieeeValue (Native.maskConstant b) (Native.maskConstant a)
  compare := rfl
  lower := by
    intro i j hi hj
    exact NativeMask.lower_mask ⟨i,hi⟩ ⟨j,hj⟩
  upper := by
    intro i j hi hj
    exact NativeMask.upper_mask ⟨i,hi⟩ ⟨j,hj⟩

#axiomsClean realMasks
#axiomsClean nativeMasks
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
