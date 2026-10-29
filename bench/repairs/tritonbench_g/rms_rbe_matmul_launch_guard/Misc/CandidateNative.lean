import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.ProgramPair
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicNativeIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmul
open KernelNative (Instructions Profile Reductions Dots)
noncomputable section

/-- The candidate's identical ordered symbolic outputs retain every existing
native instruction/profile observation, without a numerical output premise. -/
theorem candidate_native_observes (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    (candidate (KernelNative.operations R D) fp8 B p).toProgramIO.ObservesOn
      (fun _ _ => KernelNative.eval I P.schedule) (PublicIO.domain fp8 B p)
      (PublicIO.nativeFormula I P R D fp8 B p) :=
  ProgramIO.ObservesOn.of_implementsOn (candidate_implements (KernelNative.operations R D) fp8 B p) _

#axiomsClean candidate_native_observes
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PairIO
