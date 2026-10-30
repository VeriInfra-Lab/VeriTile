import bench.tritonbench_g.masked_add_cuda.Kernels
import bench.tritonbench_g.masked_add_cuda.Proofs.Mathematical_Source
import bench.tritonbench_g.vector_addition.Sources.Mathematical_Value

namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters

def parameters (grad data mask : RegionName) (n : Nat) : Parameters :=
  ⟨data, mask, grad, n, blockSize⟩

abbrev candidateKernel (storage : MaskStorage) (p : Parameters) (alpha : ℝ) :=
  kernel storage p.output p.x p.y p.n alpha p.block

theorem kernel_eq_original (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    kernel storage g p m n a B =
      VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel storage Bool.false g p m n a B := by
  cases storage <;> rfl

theorem commutedKernel_eq_original (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    commutedKernel storage g p m n a B =
      VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel storage Bool.true g p m n a B := by
  cases storage <;> rfl

theorem source_projection (storage : MaskStorage) (p : Parameters) (alpha : ℝ) :
    (candidateKernel storage p alpha).eraseDType.toAlgKernel =
      (VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_math_kernel
        p.output p.x p.y p.n alpha p.block).toAlgKernel := by
  simpa only [candidateKernel, kernel_eq_original] using
    VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_projection storage Bool.false
      p.output p.x p.y p.n alpha p.block

theorem signed_mask_channel (storage : MaskStorage) (p : Parameters) (a : ℝ) :
    (candidateKernel storage p a).surfaceBody[4]?.map
      VeriTile.Bench.TritonBenchG.MaskedAddCuda.integerMaskRead = some Bool.true := by
  cases storage <;> rfl

theorem fp32_arithmetic (storage : MaskStorage) (p : Parameters) (a : ℝ) :
    (candidateKernel storage p a).surfaceBody[8]?.map
      (VeriTile.Bench.TritonBenchG.MaskedAddCuda.arithmeticPrecision .fp32) = some Bool.true := by
  cases storage <;> rfl

#axiomsClean source_projection
#axiomsClean kernel_eq_original
#axiomsClean commutedKernel_eq_original
#axiomsClean signed_mask_channel
#axiomsClean fp32_arithmetic
end VeriTile.Bench.Repairs.MaskedAddCuda
