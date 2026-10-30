import bench.pretrain_kernels.tasks.local_qknorm_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_qknorm_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (_xs : IO.Inputs) : Prop :=
  True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1,i2,i3) =>
    let parameter0 := xs.parameter0
    let input0 := xs.input0
    let vto_copy : Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ := fun i0 i1 i2 i3 => input0 i0 i1 i2 i3
    let vpow_1 : Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ := fun i0 i1 i2 i3 => ((vto_copy i0 i1 i2 i3) * (vto_copy i0 i1 i2 i3))
    let vmean : Fin 2 → Fin 128 → Fin 16 → Fin 1 → ℝ := fun i0 i1 i2 i3 => ((∑ lane : Fin 128, vpow_1 i0 i1 i2 lane) / 128)
    let vadd : Fin 2 → Fin 128 → Fin 16 → Fin 1 → ℝ := fun i0 i1 i2 i3 => (vmean i0 i1 i2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 128 → Fin 16 → Fin 1 → ℝ := fun i0 i1 i2 i3 => (1 / Real.sqrt (vadd i0 i1 i2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul : Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy i0 i1 i2 i3 * vrsqrt i0 i1 i2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_1 : Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (parameter0 i3 * vmul i0 i1 i2 i3)
    let vto_copy_1 : Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vmul_1 i0 i1 i2 i3
    vto_copy_1 i0 i1 i2 i3

end Contract
end
end VeriTile.Bench.PretrainKernels.local_qknorm_fwd
