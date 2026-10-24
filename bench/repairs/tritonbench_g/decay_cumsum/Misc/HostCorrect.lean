import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.decay_cumsum.Kernels
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Contract
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealPublic

namespace VeriTile.Bench.Repairs.DecayCumsum.Correct
open VeriTile.Triton VeriTile.Torch PackedStorage
noncomputable section

/-- Complete guarded mathematical result for the three wrappers. The target
uses original input bytes; required padded scratch never replaces the target. -/
specification checked_whole_wrapper_correct
    (h : Call) (kind : Kind) (observed : RegionName → TensorMetadata) (scaleIsFloat : Bool)
    (contract : InputContract h kind observed scaleIsFloat) (scale : ℝ)
    (api : WholeHost.API h kind β ℝ) (targets : RegionName → Host.Target)
    (_binding : TensorBinding h kind observed targets)
    (decode : Bool → (Fin 4 → β) → ℝ) (encode : Bool → ℝ → Fin 4 → β)
    (before : Copyback.Heap β) (live : Nat → Prop)
    (ordinaryPreparation : WholeHost.PreparationLaws h kind api targets decode before live)
    (ordinaryCopy : WholeHost.CopyLaws api) :
    Host.Outcome h kind targets encode
      (fun r i=>reference h kind scale (Host.logicalData h targets decode before) r (Logical.coordinates h i)) before
      (Host.guarded h kind observed scaleIsFloat
        (WholeHost.run h kind api targets encode
          (fun p=>Run.run (RealStorage.execute h kind p.storage scale) (Grid.schedule h))
          (fun p=>LogicalStorage.crop h kind p.storage) before) before) := by
  rw [Host.guarded_eq contract]
  exact whole_wrapper_correct h kind scale api targets decode encode before live ordinaryPreparation ordinaryCopy

#axiomsClean checked_whole_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.Correct
