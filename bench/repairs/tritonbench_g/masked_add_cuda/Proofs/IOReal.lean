import bench.repairs.tritonbench_g.masked_add_cuda.IO
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.KernelRealResults

namespace VeriTile.Bench.Repairs.MaskedAddCuda.RealIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Bench.Repairs.MaskedAddCuda.Correct
open scoped VeriTile.Triton.UKernelIO
noncomputable section

theorem implements (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    Spec.Real (io storage n alpha ⊨ᵤᵣ fun _ _ vals _ j =>
      formula (vals (⟨0, by decide⟩ : Fin 3) j)
        (vals (⟨1, by decide⟩ : Fin 3) j) (vals (⟨2, by decide⟩ : Fin 3) j) alpha) := by
  unfold Spec.Real UKernelIO.RegionalImplements
  apply UKernelIO.RegionalImplements3D.intro
  · intro bounds s vals _ hi _ _
    rw [show (io storage n alpha).kernel.toAlgKernel =
      { (integer_math_kernel "grad" "data" "mask" n alpha blockSize).toAlgKernel with
        inputs := ["grad","data","mask"] } from
        source_projection storage (parameters "grad" "data" "mask" n) alpha]
    exact integer_math_kernel_traceSafe "grad" "data" "mask" n alpha blockSize bounds s
      (hi (⟨1, by decide⟩ : Fin 3)) (hi (⟨0, by decide⟩ : Fin 3))
      (hi (⟨2, by decide⟩ : Fin 3))
  · intro s vals hpins
    obtain ⟨final, he, hv, hf⟩ := integer_math_kernel_region_run
      "grad" "data" "mask" n alpha blockSize s
      (vals (⟨2, by decide⟩ : Fin 3)) (vals (⟨1, by decide⟩ : Fin 3))
      (vals (⟨0, by decide⟩ : Fin 3))
      (hpins (⟨1, by decide⟩ : Fin 3)) (hpins (⟨0, by decide⟩ : Fin 3))
      (hpins (⟨2, by decide⟩ : Fin 3))
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [io, source_execution, integer_exec] using he
    · intro o j hj
      simpa only [io, ChanTy.read, formula, if_pos hj.2] using hv j hj
    · intro r address outside _
      apply hf r address
      by_cases hr : r = "grad"
      · right
        intro j hj
        exact (outside (⟨0, by decide⟩ : Fin 1) j hj).resolve_left (by simpa using hr)
      · exact Or.inl hr

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.RealIO
