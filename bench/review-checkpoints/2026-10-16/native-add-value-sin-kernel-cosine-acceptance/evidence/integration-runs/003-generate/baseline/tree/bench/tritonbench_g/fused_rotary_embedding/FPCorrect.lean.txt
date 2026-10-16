import bench.tritonbench_g.fused_rotary_embedding.Proofs.FPQSoftware
import bench.tritonbench_g.fused_rotary_embedding.Proofs.FPVWitness

namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete selected in-place Q pair, with concrete binary32 rounding at each
multiply and add/subtract, initial-state reads, successful execution and frame. -/
specification fused_rotary_q_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPQ.Parameters)
    (s : State (FP.Value .fp32)) (hstride : 0 < p.dimensionStride) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPQ.source p) s = some final ∧
      FPQ.Post p s final (FPQ.roundedValue cfg p s) :=
  FPQ.software_correctness rest cfg p s (Nat.ne_of_gt hstride)

/-- Complete selected V chain: leader gate, context length, block table,
initial payload, readback and frame. No floating arithmetic law is required. -/
specification fused_rotary_vcache_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (p : FPV.Parameters) (s : State α)
    (hinj : FPV.active p s → Function.Injective (FPV.outputOffset p s)) :
    ∃ final, FP.Structural.exec (KcacheCopyTriton.TypedCopy.copyBackend rest)
      (FPV.source p) s = some final ∧ FPV.Post p s final :=
  FPV.correctness rest p s hinj

theorem fused_rotary_q_source_projection (p : FPQ.Parameters) :
    (FPQ.source p).eraseDType.surfaceBody = (FPQ.legacySource p).eraseDType.surfaceBody :=
  FPQ.source_projection p

theorem fused_rotary_vcache_source_projection (p : FPV.Parameters) :
    (FPV.source p).eraseDType.surfaceBody = (FPV.legacySource p).eraseDType.surfaceBody :=
  FPV.source_projection p

theorem fused_rotary_q_bit_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPQ.source FPQ.witnessParameters) FPQ.witnessInitial = some final ∧
      final.mem "q" 0 = .mk .fp32 FPQ.witnessOne ∧
      final.mem "q" 1 = .mk .fp32 FPQ.witnessOne ∧
      final.mem "q" 2 = FPQ.witnessInitial.mem "q" 2 ∧
      (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 ∧
      FPQ.witnessTiny ≠ FPQ.witnessOne :=
  FPQ.nonempty_rounding rest

theorem fused_rotary_vcache_bit_success (rest : Algebra (BitVec 32)) :
    ∃ final, FP.Structural.exec (KcacheCopyTriton.TypedCopy.copyBackend rest)
      (FPV.source FPV.witnessParameters) FPV.witnessInitial = some final ∧
      (∀ i : Fin 4, final.mem "cache" (324 + i.val) = .mk .fp32 (FPV.payload i)) ∧
      final.mem "cache" 323 = FPV.witnessInitial.mem "cache" 323 :=
  FPV.nonempty_bits rest

#axiomsClean fused_rotary_q_fp32_correctness
#axiomsClean fused_rotary_vcache_typed_correctness
#axiomsClean fused_rotary_q_source_projection
#axiomsClean fused_rotary_vcache_source_projection
#axiomsClean fused_rotary_q_bit_success
#axiomsClean fused_rotary_vcache_bit_success
end
end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding
