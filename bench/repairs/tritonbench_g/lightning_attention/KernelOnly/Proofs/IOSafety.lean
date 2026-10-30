import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.StageSafety
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.IOSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
open VeriTile Triton FP.Structural
noncomputable section

theorem forward_safety (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (forwardIO d).Inputs) (bounds : RegionName → Nat) (h : (forwardIO d).InBounds bounds x)
    (s : State ℝ) : ScheduleSafe bounds ((forwardIO d).schedule x) s := by
  apply SourceSafety.schedule_safe Real.algebra realMasks Real.dotPrimitive .forward d _ bounds
  intro pid hp
  exact forward_bounds d geometry x bounds h pid ((mem_grid _ _ _).mp hp)

theorem backward_safety (d : Host.Dimensions) (x : (backwardIO d).Inputs)
    (bounds : RegionName → Nat) (h : (backwardIO d).InBounds bounds x)
    (s : State ℝ) : ScheduleSafe bounds ((backwardIO d).schedule x) s := by
  apply (scheduleSafe_append _ _ _ _).2
  constructor
  · apply SourceSafety.schedule_safe Real.algebra realMasks Real.dotPrimitive .intra d _ bounds
    intro pid hp
    exact backward_bounds d x bounds h pid.head ((mem_grid _ _ _).mp hp).1
  · intro t ran
    apply SourceSafety.schedule_safe Real.algebra realMasks Real.dotPrimitive .inter d _ bounds
    intro pid hp
    exact backward_bounds d x bounds h pid.head ((mem_grid _ _ _).mp hp).1
#axiomsClean forward_safety
#axiomsClean backward_safety
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
