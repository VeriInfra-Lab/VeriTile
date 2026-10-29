import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelFrames
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelResults
open VeriTile.Triton Whole
noncomputable section

/-- Complete raw-state Real execution with independent dq/dk/dg formulas and frame. -/
theorem kernel_real (c : Config) (s : BlockState) (order : List (CTA c))
    (ho : order.Perm (allCTAs c)) :
    ∃ final, Whole.runGrid c order s = some final ∧ KernelReal.Result c s final :=
  KernelReal.source c s order ho

abbrev kernel_objective_gradient := @KernelObjective.objective_gradient
abbrev kernel_dq_dk_gradient := @KernelObjective.gradients_qk
abbrev kernel_dg_gradient := @KernelObjective.gradient_g
#axiomsClean kernel_real
#axiomsClean kernel_objective_gradient
#axiomsClean kernel_dq_dk_gradient
#axiomsClean kernel_dg_gradient
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelResults

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelResults
open VeriTile.Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

/-- Complete canonical quoted-source execution on raw native state. The result
contains independent dq/dk/dg word formulas and the complete regional frame. -/
theorem kernel_native (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (s : State Node) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ final, KernelNative.runQuoted (Whole.kernel (nativeConfig c)) isa route scale c order s = some final ∧
      KernelNative.Result isa route scale c s final := by
  rw [KernelNative.run_quote]
  exact KernelNative.source isa route scale c s order ho

abbrev kernel_native_source := @KernelNative.source
abbrev kernel_empty_real := @KernelFrames.real_empty
abbrev kernel_empty_native := @KernelFrames.native_empty
#axiomsClean kernel_native
#axiomsClean kernel_native_source
#axiomsClean kernel_empty_real
#axiomsClean kernel_empty_native
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelResults
