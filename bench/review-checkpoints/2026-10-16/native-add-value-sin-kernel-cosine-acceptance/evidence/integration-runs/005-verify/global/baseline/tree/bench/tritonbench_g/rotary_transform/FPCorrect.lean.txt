import bench.tritonbench_g.rotary_transform.Proofs.FPSoftware
import bench.tritonbench_g.rotary_transform.Proofs.FPShape

namespace VeriTile.Bench.TritonBenchG.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- Concrete binary32, separately rounded arithmetic for the complete already
selected paired row and both fixed/varlen metadata branches. -/
specification rotary_transform_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (k : FPRow.Kind)
    (p : FPRow.Parameters) (s : State (FP.Value .fp32)) (hstride : p.strideOutDimension ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPRow.source k p) s = some final ∧
      FPRow.Post k p s final (FPRow.roundedValue cfg k p s) :=
  FPRow.software_correctness rest cfg k p s hstride

theorem rotary_transform_source_projection (k : FPRow.Kind) (p : FPRow.Parameters) :
    (FPRow.source k p).eraseDType.surfaceBody = (FPRow.legacySource k p).eraseDType.surfaceBody :=
  FPRow.source_projection k p

theorem rotary_transform_bit_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (k : FPRow.Kind) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPRow.source k FPRow.witnessParameters) FPRow.roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real FPRow.witnessOne ∧
      final.mem "out" 1 = .mk .real FPRow.witnessOne ∧
      final.mem "out" 2 = FPRow.roundingInitial.mem "out" 2 ∧
      (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 :=
  FPRow.nonempty_rounding rest k

#axiomsClean rotary_transform_fp32_correctness
#axiomsClean rotary_transform_source_projection
#axiomsClean rotary_transform_bit_success
end
end VeriTile.Bench.TritonBenchG.RotaryTransform
