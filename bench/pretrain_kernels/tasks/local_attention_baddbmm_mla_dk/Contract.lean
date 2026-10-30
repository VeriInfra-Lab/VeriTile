import bench.pretrain_kernels.tasks.local_attention_baddbmm_mla_dk.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_attention_baddbmm_mla_dk
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
    let input0 := xs.input0
    let input1 := xs.input1
    let input2 := xs.input2
    let input3 := xs.input3
    let vtriu : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≥ i0.val + 1 then Bool.true else Bool.false)
    let vmasked_fill : Fin 2 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vtriu i2 i3 then (-338953138925153547590470800371487866880 : ℝ) else 0)
    let vbmm : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, input0 ⟨(((((i0.val * 24576) + (i1.val * 192) + lane.val) / 98304) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i1 lane * input1 ⟨(((((i0.val * 24576) + (lane.val * 128) + i2.val) / 98304) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i2 lane)
    let vmul : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm ⟨(((i0.val * 4) + i1.val)) % 8, Nat.mod_lt _ (by decide)⟩ i2 i3 * (3608439182435161/50000000000000000))
    let vadd : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul i0 i1 i2 i3 + vmasked_fill i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vto_copy : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd i0 i1 i2 i3
    let vsoftmax : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vbmm_2 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, input3 ⟨(((((i0.val * 16384) + (i1.val * 128) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ i1 ⟨((i0.val % 4)) % 4, Nat.mod_lt _ (by decide)⟩ lane * input2 ⟨(((((i0.val * 16384) + (i2.val * 128) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i2 lane)
    let vto_copy_2 : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vbmm_2 ⟨(((i0.val * 4) + i1.val)) % 8, Nat.mod_lt _ (by decide)⟩ i2 i3
    let vsoftmax_backward_data : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_2 i0 i1 i2 i3 - (∑ lane : Fin 128, vto_copy_2 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax i0 i1 i2 i3
    let vto_copy_3 : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax_backward_data i0 i1 i2 i3
    let vmul_1 : Fin 2 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_3 i0 i1 i2 i3 * (3608439182435161/50000000000000000))
    let vbmm_3 : Fin 8 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, input0 ⟨(((((i0.val * 24576) + (lane.val * 192) + i1.val) / 98304) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4)) % 4, Nat.mod_lt _ (by decide)⟩ lane i1 * vmul_1 ⟨(((((i0.val * 16384) + (lane.val * 128) + i2.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4)) % 4, Nat.mod_lt _ (by decide)⟩ lane i2)
    vbmm_3 ⟨(((i0.val * 4) + i1.val)) % 8, Nat.mod_lt _ (by decide)⟩ i3 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_attention_baddbmm_mla_dk
