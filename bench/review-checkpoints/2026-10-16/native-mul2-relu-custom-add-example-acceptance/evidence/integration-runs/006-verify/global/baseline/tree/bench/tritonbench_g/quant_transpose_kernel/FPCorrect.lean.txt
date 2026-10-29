import bench.tritonbench_g.quant_transpose_kernel.Proofs.TypedScaled
import bench.tritonbench_g.quant_transpose_kernel.Proofs.TypedShape
namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel.FPCorrect
open VeriTile.Triton FP.Structural TypedScaled
noncomputable section

/-- The accepted pre-rounding stage, with two separately rounded fp32
multiplications, exact fp32 literal127, masked transposed stores and full frame.
This does not claim the llrint/int8 output or the grouped-grid launch pipeline. -/
specification quant_transpose_scaled_store_fp32_correctness
    (rest : Algebra Word) (cfg : FP.Config) (s : State Word)
    (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat)
    (hOutInj : Function.Injective (outputOffset s sbn sbm BM BN)) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source A AbsmaxInv B sam san sbn sbm M N BM BN) s = some t ∧
      (∀ i, TypedScaled.active s M N BM BN i →
        t.mem B (outputOffset s sbn sbm BM BN i) = .mk .real (value cfg s A AbsmaxInv sam san BM BN i)) ∧
      (∀ r o, (r ≠ B ∨ ∀ i, TypedScaled.active s M N BM BN i → o ≠ outputOffset s sbn sbm BM BN i) →
        t.mem r o = s.mem r o) :=
  TypedScaled.source_correct rest cfg s A AbsmaxInv B sam san sbn sbm M N BM BN hOutInj

#axiomsClean quant_transpose_scaled_store_fp32_correctness
end
end VeriTile.Bench.TritonBenchG.QuantTransposeKernel.FPCorrect
