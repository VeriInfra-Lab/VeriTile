import bench.repairs.tritonbench_g.chunk_retention.Proofs.ScaleFirstPair

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
open VeriTile Triton FP.Structural KernelPacket KernelStaging
open scoped VeriTile.Spec
noncomputable section

/-- Existing scale-first final multiply candidate, on the entire selected
four-stage interface. Every source dot/FMA chain, conversion, mask and output
window is retained; only forward-output's final FP32 operands commute. The
fixed software multiplication theorem covers all words, including nonfinite
values. The scalar/native binding remains the original explicit boundary. -/
specification source_candidate_equivalence
    (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ → BitVec 32)
    (literals : LiteralLaws (wordOps (Numeric.nativeISA ports)) literal)
    (a : Call) (p grid : Nat → Nat) (legal : PublicIO.Legal a p grid) :
    PairIO.original (wordOps (Numeric.nativeISA ports)) literal a p grid ≡[[]]
      PairIO.scaleFirst (wordOps (Numeric.nativeISA ports)) literal a p grid := by
  exact Spec.FloatingPoint.ofStructural rfl rfl
    (PairIO.equivalent _ _ literals rfl (ScaleFirstSource.native_mul_comm u ports laws) a p grid legal)

#axiomsClean source_candidate_equivalence
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
