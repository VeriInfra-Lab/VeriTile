import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.ScaleFirstPair

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.FPEquiv
open VeriTile Triton FP.Structural KernelPacket KernelNative
open scoped VeriTile.Spec
noncomputable section

/-- Complete actual four-stage packet pair for the existing scale-first final
FP32 multiply. The original arbitrary PID/all-word domain, all eight outputs,
source safety and exact raw exterior frames are retained. Only forward-output
changes; its dot/seed/FMA expression is unchanged. -/
specification source_candidate_equivalence (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ → Word)
    (constants : S.LiteralLaws (sharedOps (wordOps (Numeric.referenceISA u))) literal) (a : Call) :
    PairIO.original (Numeric.nativeISA ports) literal a ≡[[]]
      PairIO.scaleFirst (Numeric.nativeISA ports) literal a := by
  have nativeConstants : S.LiteralLaws (sharedOps (wordOps (Numeric.nativeISA ports))) literal := by
    simpa only [Numeric.isa_eq u ports laws] using constants
  exact Spec.FloatingPoint.ofStructural rfl rfl
    (PairIO.equivalent _ _ nativeConstants (PairIO.native_mul_comm u ports laws) a)

#axiomsClean source_candidate_equivalence
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.FPEquiv
