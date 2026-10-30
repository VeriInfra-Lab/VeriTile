import bench.pretrain_kernels.tasks.local_fused_hc_post_bda_dbias.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_dbias
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
  fun xs i0 =>
    let input0 := xs.input0
    let input1 := xs.input1
    let input2 := xs.input2
    let vto_copy_2 : Fin 2 → Fin 4096 → Fin 2048 → ℝ := fun i0 i1 i2 => (if input1 i0 i1 i2 then 1 else 0)
    let vto_copy_5 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => input2 i0 i1 i2 i3
    let vmul_3 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (vto_copy_5 i0 i1 i2 i3 * input0 i0 i1 i2)
    let vsum_1 : Fin 2 → Fin 4096 → Fin 1 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (∑ lane : Fin 4, vmul_3 i0 i1 lane i3)
    let vmul_4 : Fin 2 → Fin 4096 → Fin 2048 → ℝ := fun i0 i1 i2 => (vsum_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 * (1388888888888889/1250000000000000))
    let vmul_5 : Fin 2 → Fin 4096 → Fin 2048 → ℝ := fun i0 i1 i2 => (vmul_4 i0 i1 i2 * vto_copy_2 i0 i1 i2)
    let vsum_2 : Fin 1 → Fin 1 → Fin 2048 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 8192, vmul_5 ⟨(((lane.val / 4096) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 4096)) % 4096, Nat.mod_lt _ (by decide)⟩ i2)
    let vto_copy_6 : Fin 2048 → ℝ := fun i0 => vsum_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0
    vto_copy_6 i0

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_dbias
