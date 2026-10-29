import bench.pretrain_kernels.tasks.local_fa4_mla_decode.Contract

namespace VeriTile.Bench.PretrainKernels.local_fa4_mla_decode
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1,i2,i3) =>
    let input0 := xs.input0
    let input1 := xs.input1
    let input2 := xs.input2
    let input3 := xs.input3
    let vcat : Fin 4 → Fin 1 → Fin 8 → Fin 576 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 64 then input0 i0 i1 i2 ⟨((i3.val - 0)) % 64, Nat.mod_lt _ (by decide)⟩ else input2 i0 i1 i2 ⟨((i3.val - 64)) % 512, Nat.mod_lt _ (by decide)⟩)
    let vcat_1 : Fin 4 → Fin 256 → Fin 1 → Fin 576 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 64 then input1 i0 i1 i2 ⟨((i3.val - 0)) % 64, Nat.mod_lt _ (by decide)⟩ else input3 i0 i1 i2 ⟨((i3.val - 64)) % 512, Nat.mod_lt _ (by decide)⟩)
    let vbmm : Fin 32 → Fin 1 → Fin 256 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 576, vcat ⟨(((((i0.val * 576) + (i1.val * 576) + lane.val) / 4608) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val + i1.val) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane * vcat_1 ⟨(((((i0.val * 147456) + (lane.val * 256) + i2.val) / 1179648) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ lane)
    let vmul : Fin 4 → Fin 8 → Fin 1 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (vbmm ⟨(((i0.val * 8) + i1.val)) % 32, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i3 * (5208333333333333/125000000000000000))
    let vto_copy : Fin 4 → Fin 8 → Fin 1 → Fin 256 → ℝ := fun i0 i1 i2 i3 => vmul i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i3
    let vsoftmax : Fin 4 → Fin 8 → Fin 1 → Fin 256 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy i0 i1 i2 i3) / (∑ lane : Fin 256, Real.exp (vto_copy i0 i1 i2 ⟨(lane.val) % 256, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_1 : Fin 4 → Fin 8 → Fin 1 → Fin 256 → ℝ := fun i0 i1 i2 i3 => vsoftmax i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i3
    let vbmm_1 : Fin 32 → Fin 1 → Fin 512 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vto_copy_1 ⟨(((((i0.val * 256) + (i1.val * 256) + lane.val) / 2048) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val + i1.val) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ lane * input3 ⟨(((((i0.val * 131072) + (lane.val * 512) + i2.val) / 1048576) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ lane ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2)
    vbmm_1 ⟨(((i0.val * 8) + i2.val + i1.val)) % 32, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i3)

end
end VeriTile.Bench.PretrainKernels.local_fa4_mla_decode
