import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.ProgramPair
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNativeProfile

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmul
open scoped VeriTile.Spec
noncomputable section

/-- Complete original and existing output-address candidate grids. The source
retains masked K tails, optional rotary, FP32 arithmetic, FP16/E5 storage, native
power/dot/reduction order, every output and the exterior cell frame. Signed-byte
storage is the original domain; all floating inputs remain admitted. -/
specification source_candidate_equivalence (R : KernelNative.Reductions) (D : KernelNative.Dots)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    PairIO.original (KernelNative.operations R D) fp8 B p ≡[[]]
      PairIO.indexCommuted (KernelNative.operations R D) fp8 B p := by
  cases fp8 <;>
    exact ProgramIO.Implementation.of_equivalentOn _
      (PairIO.equivalent (KernelNative.operations R D) _ B p)

#axiomsClean source_candidate_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.FPEquiv
