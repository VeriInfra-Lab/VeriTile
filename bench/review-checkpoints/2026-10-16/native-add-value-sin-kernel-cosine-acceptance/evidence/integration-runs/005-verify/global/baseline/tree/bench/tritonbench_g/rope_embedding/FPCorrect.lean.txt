import bench.tritonbench_g.rope_embedding.Proofs.FPWitness
import bench.tritonbench_g.rope_embedding.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.RopeEmbedding
open VeriTile.Triton FP.Structural
noncomputable section

/-- All four accepted half-store stages, including actual backward fneg,
register casts, initial-state numerical reference, success, readback and frame. -/
specification rope_embedding_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (backward second : Bool)
    (p : FPHalves.Parameters) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (FPHalves.source backward second p) s = some final ∧
      FPHalves.Post second p s final (FPHalves.roundedValue cfg backward second p s) :=
  FPHalves.software_correctness rest cfg backward second p s

theorem rope_embedding_source_projection (backward second : Bool) (p : FPHalves.Parameters) :
    (FPHalves.source backward second p).eraseDType.surfaceBody =
      (FPHalves.legacySource backward second p).eraseDType.surfaceBody :=
  FPHalves.source_projection backward second p

theorem rope_embedding_bit_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (backward second : Bool) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPHalves.source backward second FPHalves.witnessParameters) FPHalves.witnessInitial = some final ∧
      final.mem "q" (if second then 1 else 0) = .mk .fp32 (FPHalves.expected backward second) ∧
      final.mem "q" 2 = FPHalves.witnessInitial.mem "q" 2 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 :=
  FPHalves.nonempty_rounding rest backward second

theorem rope_embedding_backward_signed_zero_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPHalves.source Bool.true Bool.true FPHalves.witnessParameters) FPHalves.zeroInitial = some final ∧
      final.mem "q" 1 = .mk .fp32 FPHalves.negativeZero ∧ FPHalves.negativeZero ≠ FP.Value.zero .fp32 :=
  FPHalves.backward_signed_zero rest

#axiomsClean rope_embedding_fp32_correctness
#axiomsClean rope_embedding_source_projection
#axiomsClean rope_embedding_bit_success
#axiomsClean rope_embedding_backward_signed_zero_success
end
end VeriTile.Bench.TritonBenchG.RopeEmbedding
