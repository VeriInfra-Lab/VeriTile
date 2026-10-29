import bench.pretrain_kernels.tasks.local_mla_rope_fused_dkv.Contract

namespace VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dkv
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
    let vcat_8 : Fin 2 → Fin 12 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 16 then input0 i0 i1 i2 ⟨(((i3.val - 0) + 24)) % 64, Nat.mod_lt _ (by decide)⟩ else input0 i0 i1 i2 ⟨(((i3.val - 16) + 48)) % 64, Nat.mod_lt _ (by decide)⟩)
    vcat_8 i0 i1 i2 i3)

end
end VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dkv
