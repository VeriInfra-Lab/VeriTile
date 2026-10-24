import bench.tritonbench_g.matmul_tma.Proofs.FPSoftware
import bench.tritonbench_g.matmul_tma.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Entire accepted single-tile program, arbitrary dimensions/strides/input
payloads/aliases, and both output branches. Only the original output-layout
injectivity is needed. The dot implementation remains an explicit parameter. -/
specification matmul_tma_fp_backend_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : Dot α) (p : Parameters) (half : Bool) (s : State α)
    (hi : Function.Injective (FPValue.cOffset p)) :
    ∃ t, FP.Structural.exec (backend M D) (source p half) s = some t ∧
      Post p half s t (outputValues M D p half s) :=
  correctness M D p half s hi

/-- Direct binary16 output rounding or bit-preserving binary32 output, with
an explicit dot primitive. This is not a fully closed numerical dot contract. -/
specification matmul_tma_fp_cast_correctness (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot Word) (p : Parameters) (half : Bool) (s : State Word)
    (hi : Function.Injective (FPValue.cOffset p)) :
    ∃ t, FP.Structural.exec (backend (software rest cfg) D) (source p half) s = some t ∧
      Post p half s t (bitOutput cfg D p half s) := by
  simpa only [output_software] using correctness (software rest cfg) D p half s hi

/-- Closed software interpretation with explicitly sequential, separately
rounded binary32 dot accumulation and direct output conversion. Native dot
ordering, fusion, precision, subnormal policy and storage/launch binding remain
separate obligations; no hardware equivalence is asserted. -/
specification matmul_tma_fp_software_correctness (rest : Algebra Word)
    (p : Parameters) (half : Bool) (s : State Word)
    (hi : Function.Injective (FPValue.cOffset p)) :
    ∃ t, FP.Structural.exec (backend (software rest .ieeeValue) sequentialDot)
      (source p half) s = some t ∧
      Post p half s t (bitOutput .ieeeValue sequentialDot p half s) :=
  matmul_tma_fp_cast_correctness rest .ieeeValue sequentialDot p half s hi

#axiomsClean matmul_tma_fp_backend_correctness
#axiomsClean matmul_tma_fp_cast_correctness
#axiomsClean matmul_tma_fp_software_correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPCorrect
