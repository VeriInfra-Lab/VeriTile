import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourceControl
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.LegacySurface

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.SourcePreservation
open VeriTile.Triton FP.Structural PairConnection
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
variable {A : Type} [Inhabited A]

/-- Semantic connection to the historical presentation. The source AST differs
at the early-return condition and the constexpr stride expression. Every
arithmetic operation, cast, memory event and failure is retained. -/
theorem original_code (M : Algebra A) (h : Call) :
    CodeEq M (Kernels.source h).surfaceBody (LegacySurface.source h).surfaceBody := by
  unfold Kernels.source LegacySurface.source
  cases h.storage <;>
    unfold Kernels.originalFP16 Kernels.originalBF16 ComputeKernel.surfaceBody
  all_goals
    repeat' first
      | exact code_refl M _
      | exact step_refl M _
      | apply code_cons
      | apply SourceControl.early_return
      | exact SourceControl.constant_stride M _ _

/-- Successful or failed execution is identical; this is proved for every
algebra and state, without a numerical law or a whole-kernel premise. -/
theorem original (M : Algebra A) (h : Call) (s : State A) :
    exec M (Kernels.source h) s = exec M (LegacySurface.source h) s := by
  have same := (original_code M h).run s
  cases e : h.storage <;>
    simpa only [Kernels.source,LegacySurface.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,
      e,FP.Structural.exec,ComputeKernel.surfaceBody] using same

theorem original_safe (M : Algebra A) (h : Call) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (Kernels.source h) s ↔ TraceSafe M bounds (LegacySurface.source h) s :=
  (original_code M h).safe bounds s

end
#axiomsClean original_code
#axiomsClean original
#axiomsClean original_safe
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.SourcePreservation
