import bench.tritonbench_g.context_attn_mistral.Proofs.LogicalMath
import bench.tritonbench_g.context_attn_mistral.Proofs.RealImplementation

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section

/-- Tensor coordinates and all retained storage cells yield exactly the old target. -/
theorem target_transport (c : Host.Call) (capacity : RegionBounds) :
    (fun raw => (outputEquiv c).symm
      (TensorMath.finiteSentinelAttention c capacity (inputEquiv c capacity raw))) =
      RealInterface.formula c capacity := rfl

theorem metadata_pullback (c : Host.Call) (capacity : RegionBounds) :
    (fun raw => Metadata c capacity (inputEquiv c capacity raw)) =
      RealInterface.Metadata c capacity := by
  funext raw
  change RealInterface.Metadata c capacity
    ((inputEquiv c capacity).symm (inputEquiv c capacity raw)) = _
  rw [Equiv.symm_apply_apply]

/-- Same ordinary grid, metadata domain, complete observations, safety and frame. -/
theorem logical_iff (c : Host.Call) (capacity : RegionBounds) :
    (attentionIO c capacity).ImplementsOn (Metadata c capacity)
      (TensorMath.finiteSentinelAttention c capacity) ↔
      (RealInterface.programIO c capacity).ImplementsOn
        (RealInterface.Metadata c capacity) (RealInterface.formula c capacity) := by
  change (RealInterface.programIO c capacity).ImplementsOn
    (fun raw => Metadata c capacity (inputEquiv c capacity raw))
    (fun raw => (outputEquiv c).symm
      (TensorMath.finiteSentinelAttention c capacity (inputEquiv c capacity raw))) ↔ _
  exact Iff.of_eq (congrArg₂ (fun domain formula =>
    (RealInterface.programIO c capacity).ImplementsOn domain formula)
      (metadata_pullback c capacity) (target_transport c capacity))

theorem logical_correct (c : Host.Call) (capacity : RegionBounds)
    (geometry : RealInterface.Geometry c capacity) :
    (attentionIO c capacity).ImplementsOn (Metadata c capacity)
      (TensorMath.finiteSentinelAttention c capacity) :=
  (logical_iff c capacity).mpr (RealInterface.implements c capacity geometry)

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData
