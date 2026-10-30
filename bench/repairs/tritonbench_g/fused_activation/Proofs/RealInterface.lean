import bench.repairs.tritonbench_g.fused_activation.IO
import bench.repairs.tritonbench_g.fused_activation.Proofs.Mathematics
import VeriTile.Triton.Memory.RegionalGroupedKernelIO

namespace VeriTile.Bench.Repairs.FusedActivation.RealInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FusedActivation
open RealIO
open scoped VeriTile.Triton.GroupedMasked2DKernelIO
noncomputable section

/-- Canonical-source correctness from the public logical input channels.
The regional judgment proves source access safety, successful execution, every
active mathematical output and the exact exterior frame. It retains coherent
read-only aliases without imposing disjoint flat allocation. -/
theorem regional_correct (x b input : RegionName)
    (W N B : Nat) (a : Bool) :
    Spec.Real (activationIO x b input W N B a ⊨ᵣ fun _ _ xs _ j =>
      formula a (xs (⟨0, by decide⟩ : Fin 3) j)
        (xs (⟨1, by decide⟩ : Fin 3) j) (xs (⟨2, by decide⟩ : Fin 3) j)) := by
  refine GroupedMasked2DKernelIO.RegionalImplements.intro _ ?_ ?_ ?_
  · intro o
    simp [activationIO]
  · intro bounds s hr _hw
    change Kernel.TraceSafe bounds (kernel x b input W N B (1/2) a).toAlgKernel s
    rw [Projection.original]
    exact fused_add_mul_activation_kernel_traceSafe x b input W N B (1/2) a bounds s
      (fun j hj => hr (⟨0, by decide⟩ : Fin 3) j hj)
      (fun j hj => hr (⟨1, by decide⟩ : Fin 3) j hj)
      (fun j hj => hr (⟨2, by decide⟩ : Fin 3) j hj)
  · intro s xs hx
    obtain ⟨t, he, hv, hf⟩ := fused_add_mul_activation_kernel_region_run
      x b input W N B (1/2) a s
      (fun j => xs (⟨0, by decide⟩ : Fin 3) j)
      (fun j => xs (⟨1, by decide⟩ : Fin 3) j)
      (fun j => xs (⟨2, by decide⟩ : Fin 3) j)
      (fun j hj => hx (⟨0, by decide⟩ : Fin 3) j hj)
      (fun j hj => hx (⟨1, by decide⟩ : Fin 3) j hj)
      (fun j hj => hx (⟨2, by decide⟩ : Fin 3) j hj)
    refine ⟨t, ?_, ?_, fun r o outside => hf r o (fun j hj => outside (⟨0, by decide⟩ : Fin 1) j hj)⟩
    · change exec (kernel x b input W N B (1/2) a).toAlgKernel s = some t
      rw [Projection.original]
      exact he
    · intro o j hj
      exact (hv j hj).trans (Mathematics.formula_eq a _ _ _)

#axiomsClean regional_correct
end
end VeriTile.Bench.Repairs.FusedActivation.RealInterface
