import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeForward
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeBackward
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeGradient
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.RawNative
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelPacket (Call)
open KernelNative
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
noncomputable section

/-- Every raw kernel stage executes successfully; all native observations
and the complete exterior frame use the same original word memory. Existing
scalar primitive and literal-conversion boundaries remain explicit. -/
theorem kernel_native (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (_constants : S.LiteralLaws (sharedOps (wordOps (Numeric.referenceISA u))) literal) (a : Call)
    (m : RegionName→Nat→Word) :
    ∃ t,FP.Structural.exec (engine (withDot model (dotModel a))) a.kernel (a.state (input m))=some t ∧
      (∀ r off,observe (Numeric.nativeISA ports) literal t.mem r off=
        target (Numeric.referenceISA u) literal a m r off) ∧
      (∀ r off,¬KernelPacket.Footprint a r off →
        observe (Numeric.nativeISA ports) literal t.mem r off=m r off) :=
  KernelNative.correct u ports laws literal a m

export KernelNative (forward_target output_target backward_target gradient_target)
#axiomsClean kernel_native
#axiomsClean forward_target
#axiomsClean output_target
#axiomsClean backward_target
#axiomsClean gradient_target
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.RawNative
