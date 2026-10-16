import bench.tritonbench_g.rotary_transform_ops.Proofs.FPStages

namespace VeriTile.Bench.TritonBenchG.RotaryTransformOps
open VeriTile.Triton FP.Structural
noncomputable section

specification rotary_transform_ops_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (k : FPStages.Kind)
    (p : FPStages.Parameters) (s : State (FP.Value .fp32)) (hstride : p.strideOutDimension ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPStages.source k p) s = some final ∧
      RotaryTransform.FPRow.Post k p s final (RotaryTransform.FPRow.roundedValue cfg k p s) :=
  FPStages.software_correctness rest cfg k p s hstride

theorem rotary_transform_ops_source_projection (k : FPStages.Kind) (p : FPStages.Parameters) :
    (FPStages.source k p).eraseDType.surfaceBody = (FPStages.legacySource k p).eraseDType.surfaceBody :=
  FPStages.source_projection k p

theorem rotary_transform_ops_bit_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (k : FPStages.Kind) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPStages.source k RotaryTransform.FPRow.witnessParameters)
      RotaryTransform.FPRow.roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real RotaryTransform.FPRow.witnessOne ∧
      final.mem "out" 1 = .mk .real RotaryTransform.FPRow.witnessOne ∧
      final.mem "out" 2 = RotaryTransform.FPRow.roundingInitial.mem "out" 2 ∧
      (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 :=
  RotaryTransform.FPRow.nonempty_rounding rest k

#axiomsClean rotary_transform_ops_fp32_correctness
#axiomsClean rotary_transform_ops_source_projection
#axiomsClean rotary_transform_ops_bit_success
end
end VeriTile.Bench.TritonBenchG.RotaryTransformOps
