import bench.pretrain_kernels.tasks.local_fa4_mla_prefill.Contract

namespace VeriTile.Bench.PretrainKernels.local_fa4_mla_prefill
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
    let vcat : Fin 2 → Fin 128 → Fin 8 → Fin 576 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 64 then input0 i0 i1 i2 ⟨((i3.val - 0)) % 64, Nat.mod_lt _ (by decide)⟩ else input2 i0 i1 i2 ⟨((i3.val - 64)) % 512, Nat.mod_lt _ (by decide)⟩)
    let vcat_1 : Fin 2 → Fin 128 → Fin 1 → Fin 576 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 64 then input1 i0 i1 i2 ⟨((i3.val - 0)) % 64, Nat.mod_lt _ (by decide)⟩ else input3 i0 i1 i2 ⟨((i3.val - 64)) % 512, Nat.mod_lt _ (by decide)⟩)
    let vtriu : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≥ i0.val + 1 then Bool.true else Bool.false)
    let vmasked_fill : Fin 2 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vtriu i2 i3 then (-338953138925153547590470800371487866880 : ℝ) else 0)
    let vbmm : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 576, vcat ⟨(((((i0.val * 73728) + (i1.val * 576) + lane.val) / 589824) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ i1 ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane * vcat_1 ⟨(((((i0.val * 73728) + (lane.val * 128) + i2.val) / 589824) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ i2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ lane)
    let vmul : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 i3 * (5208333333333333/125000000000000000))
    let vadd : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul i0 i1 i2 i3 + vmasked_fill i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vto_copy : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd i0 i1 i2 i3
    let vsoftmax : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_1 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax i0 i1 i2 i3
    let vbmm_1 : Fin 16 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_1 ⟨(((((i0.val * 16384) + (i1.val * 128) + lane.val) / 131072) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 lane * input3 ⟨(((((i0.val * 65536) + (lane.val * 512) + i2.val) / 524288) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ lane ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2)
    vbmm_1 ⟨(((i0.val * 8) + i2.val)) % 16, Nat.mod_lt _ (by decide)⟩ i1 i3)

end
end VeriTile.Bench.PretrainKernels.local_fa4_mla_prefill
