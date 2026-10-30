import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.IOReal
import bench.repairs.tritonbench_g.masked_add_cuda.IO
namespace VeriTile.Bench.Repairs.MaskedAddCuda.ReaderProof
open VeriTile Triton ReaderIO
open scoped LogicalIO
noncomputable section

/-- Exact old/new judgment equivalence over all regional pids, input channels,
actual source execution, access safety and the complete data-dependent frame. -/
theorem judgment_iff (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    (view storage n alpha ⊨ fun xs _ _ j => if xs.mask j = 0 then xs.gradient j + xs.data j * alpha else xs.gradient j) ↔
      (RealIO.io storage n alpha).RegionalImplements (fun _ _ vals _ j =>
        Correct.formula (vals (0 : Fin 3) j) (vals (1 : Fin 3) j)
          (vals (2 : Fin 3) j) alpha) := by
  apply Iff.of_eq
  apply congrArg (UKernelIO.RegionalImplements (RealIO.io storage n alpha))
  funext p₀ p₁ vals o j
  fin_cases o
  rfl

theorem correct (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    (view storage n alpha ⊨ fun xs _ _ j => if xs.mask j = 0 then xs.gradient j + xs.data j * alpha else xs.gradient j) :=
  (judgment_iff storage n alpha).mpr (RealIO.implements storage n alpha)
end
end VeriTile.Bench.Repairs.MaskedAddCuda.ReaderProof
