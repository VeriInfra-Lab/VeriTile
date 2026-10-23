import bench.repairs.tritonbench_g.fused_activation.Proofs.RealInterface

/-! Optional flat-allocation corollary. Regional IO remains the primary entry. -/
namespace VeriTile.Bench.Repairs.FusedActivation.Flat
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FusedActivation
open RealIO
open scoped VeriTile.Triton.GroupedMasked2DKernelIO
noncomputable section

/-- The same certificate also retains the optional disjoint flat-memory
endpoint. This is a corollary of the primary regional specification. -/
theorem fused_activation_kernel_flat_correctness (x b input : RegionName)
    (W N B : Nat) (a : Bool) (_weightsPositive : 0 < W) :
    Spec.Real (activationIO x b input W N B a ⊨ fun _ _ xs _ j =>
      formula a (xs (⟨0, by decide⟩ : Fin 3) j)
        (xs (⟨1, by decide⟩ : Fin 3) j) (xs (⟨2, by decide⟩ : Fin 3) j)) :=
  GroupedMasked2DKernelIO.RegionalImplements.toFlat
    (RealInterface.regional_correct x b input W N B a)
    (by
      change Kernel.FlattenOk (kernel x b input W N B (1/2) a).toAlgKernel
      rw [Projection.original]
      exact fused_add_mul_activation_kernel_flattenOk x b input W N B (1/2) a)

#axiomsClean fused_activation_kernel_flat_correctness
end
end VeriTile.Bench.Repairs.FusedActivation.Flat
