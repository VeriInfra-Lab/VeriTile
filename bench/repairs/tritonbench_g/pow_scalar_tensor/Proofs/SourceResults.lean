import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Source
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeReal
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section

/-- Real precision erasure interprets the retained source cast explicitly. -/
theorem source_real_scalar (rest : Algebra ℝ) (base x : ℝ) :
    scalarResult (realBackend rest) base x = Real.rpow base x := by
  simp [scalarResult,WholeTile.scalarResult,WholeTile.scalarBase,sourceCastBridge,realBackend]

/-- The existing word backend defines same-format casts as identity. This is
specific to that interpreter, not a premise on an arbitrary floating algebra. -/
theorem source_fp32_scalar (rest : Algebra Float32Bits) (baseWord : Float32Bits)
    (base : ℝ) (exponent : Float32Bits) :
    scalarResult (fp32Backend rest baseWord) base exponent =
      rest.binary (some .fp32) .real .pow baseWord exponent := rfl

#axiomsClean source_real_scalar
#axiomsClean source_fp32_scalar
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked
