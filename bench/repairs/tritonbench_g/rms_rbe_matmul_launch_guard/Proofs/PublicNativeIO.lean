import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicSourceIO
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNativeProfile
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
open KernelNative (Term Word Instructions Profile Reductions Dots)
noncomputable section

/-- Complete native observation of the independent ordered expression graph.
The scalar/conversion/K8 instruction and compiler profile parameters retain
their existing primitive meanings; every input term remains admitted. -/
def nativeFormula (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (fp8 : Bool) (B : Nat) (p : Parameters) (xs : ChannelValues Term (inputs fp8 B p)) :
    ChannelValues Word (outputs fp8 B p) :=
  fun o j => mapValue (KernelNative.eval I P.schedule) (outputs fp8 B p)[o].dtype
    (formula (KernelNative.operations R D) fp8 B p xs o j)

theorem native_observes (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (fp8 : Bool) (B : Nat) (p : Parameters) :
    (io (KernelNative.operations R D) fp8 B p).ObservesOn
      (fun _ _ => KernelNative.eval I P.schedule) (domain fp8 B p)
      (nativeFormula I P R D fp8 B p) :=
  ProgramIO.ObservesOn.of_implementsOn (implements (KernelNative.operations R D) fp8 B p) _

#specNonCircular nativeFormula avoiding [FP.Structural.exec,KernelSource.inputs,KernelGrid.launch]
#axiomsClean native_observes
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
